//! VM-local many-to-many event routing. Connections are GC edges, not Arc
//! cycles. A receiver owns one handler per subscribed channel; events are not
//! buffered, they are handed to that handler as a microtask.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PubSubNative {
    EventNew, EventData, EmitterNew, ChannelNew, ReceiverNew,
    PublishTo, UnpublishFrom, SubscribeFrom, UnsubscribeFrom, Emit,
    ChannelCount, EmitterCount, ReceiverCount, Close, Closed,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NativeSubscription {
    channel: Value,
    handler: Value,
}

#[derive(Debug, Default)]
pub(super) struct NativeReceiverState {
    subscriptions: Vec<NativeSubscription>,
    /// Handler microtasks queued but not yet run. A receiver with in-flight
    /// deliveries cannot be retyped.
    deliveries: usize,
}

impl NativeReceiverState {
    fn handler_for(&self, channel: Value) -> Option<Value> {
        self.subscriptions.iter().find(|subscription| subscription.channel == channel).map(|subscription| subscription.handler)
    }
}

#[derive(Debug)]
pub(super) enum NativeEndpointData {
    Event(Value),
    Emitter(Vec<Value>),
    Channel { emitters: Vec<Value>, receivers: Vec<Value> },
    Receiver(Box<NativeReceiverState>),
}

#[derive(Debug)]
pub(super) struct NativeEndpointState {
    pub(super) payload_type: Option<Type>,
    pub(super) closed: bool,
    pub(super) data: NativeEndpointData,
}

impl NativeEndpointState {
    pub(super) fn visit_values(&self, mut visit: impl FnMut(Value)) {
        match &self.data {
            NativeEndpointData::Event(value) => visit(*value),
            NativeEndpointData::Emitter(channels) => channels.iter().copied().for_each(visit),
            NativeEndpointData::Channel { emitters, receivers } => {
                emitters.iter().chain(receivers).copied().for_each(visit);
            }
            NativeEndpointData::Receiver(receiver) => {
                for subscription in &receiver.subscriptions {
                    visit(subscription.channel);
                    visit(subscription.handler);
                }
            }
        }
    }

    pub(super) fn external_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + match &self.data {
            NativeEndpointData::Event(_) => 0,
            NativeEndpointData::Emitter(channels) => channels.capacity() * std::mem::size_of::<Value>(),
            NativeEndpointData::Channel { emitters, receivers } =>
                (emitters.capacity() + receivers.capacity()) * std::mem::size_of::<Value>(),
            NativeEndpointData::Receiver(receiver) => std::mem::size_of::<NativeReceiverState>()
                + receiver.subscriptions.capacity() * std::mem::size_of::<NativeSubscription>(),
        }
    }

    // Retyping a live network would silently invalidate its other endpoints or
    // queued handler calls. Install an endpoint's generic contract before using it.
    pub(super) fn can_set_payload_type(&self) -> bool {
        match &self.data {
            NativeEndpointData::Event(_) => true,
            NativeEndpointData::Emitter(channels) => channels.is_empty(),
            NativeEndpointData::Channel { emitters, receivers } => emitters.is_empty() && receivers.is_empty(),
            NativeEndpointData::Receiver(receiver) => receiver.subscriptions.is_empty() && receiver.deliveries == 0,
        }
    }
}

pub(super) struct NativeEndpointCell {
    state: std::cell::UnsafeCell<NativeEndpointState>,
}

impl NativeEndpointCell {
    fn new(data: NativeEndpointData, payload_type: Option<Type>) -> Self {
        Self { state: std::cell::UnsafeCell::new(NativeEndpointState { data, payload_type, closed: false }) }
    }
    pub(super) unsafe fn get(&self) -> &NativeEndpointState { unsafe { &*self.state.get() } }
    pub(super) unsafe fn get_mut(&self) -> &mut NativeEndpointState { unsafe { &mut *self.state.get() } }
}

// Only the owning VM mutates an endpoint. GC traces it with mutators stopped;
// endpoint Values are explicitly rejected at concurrent transfer boundaries.
unsafe impl Send for NativeEndpointCell {}
unsafe impl Sync for NativeEndpointCell {}
impl fmt::Debug for NativeEndpointCell {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result { out.write_str("NativeEndpointCell(..)") }
}

impl Interpreter<'_> {
    fn endpoint(&self, value: Value) -> Result<Arc<NativeEndpointCell>, RuntimeError> {
        match self.heap.get(value) {
            Some(HeapObject::Event { state } | HeapObject::Emitter { state }
                | HeapObject::Channel { state } | HeapObject::Receiver { state }) => Ok(state.clone()),
            _ => Err(RuntimeError::new("expected an Event, Emitter, Channel or Receiver")),
        }
    }

    fn endpoint_changed(&mut self, value: Value) {
        self.heap.remember_owner(value);
        let bytes = self.heap.get(value).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(value, bytes);
    }

    fn allocate_event(&mut self, data: Value, payload_type: Option<Type>) -> Value {
        self.heap.alloc(HeapObject::Event {
            state: Arc::new(NativeEndpointCell::new(NativeEndpointData::Event(data), payload_type)),
        })
    }

    /// Connect an Emitter (`handler == None`) or a Receiver (`Some(handler)`)
    /// to a Channel. Re-subscribing an already connected receiver replaces its
    /// handler for that channel.
    fn endpoint_connect(&mut self, owner: Value, channel: Value, handler: Option<Value>) -> Result<Value, RuntimeError> {
        let publishing = handler.is_none();
        if let Some(handler) = handler {
            if handler == Value::NULL {
                return Err(RuntimeError::new("Receiver.subscribe_from(channel, handler) requires a handler function"));
            }
            self.receiver_handler_function(handler)?;
        }
        let owner_cell = self.endpoint(owner)?;
        let channel_cell = self.endpoint(channel)?;
        let owner_state = unsafe { owner_cell.get() };
        let channel_state = unsafe { channel_cell.get() };
        if !matches!(channel_state.data, NativeEndpointData::Channel { .. }) {
            return Err(RuntimeError::new("connection target must be a Channel"));
        }
        let already = match (&owner_state.data, publishing) {
            (NativeEndpointData::Emitter(channels), true) => channels.contains(&channel),
            (NativeEndpointData::Receiver(receiver), false) => receiver.handler_for(channel).is_some(),
            _ => return Err(RuntimeError::new("invalid connection endpoint")),
        };
        if owner_state.closed || channel_state.closed {
            return Err(RuntimeError::new("cannot connect a closed endpoint"));
        }
        if owner_state.payload_type != channel_state.payload_type {
            let describe = |name: &str, ty: &Option<Type>| match ty {
                Some(ty) => format!("{name}<{}>", self.type_display(ty)),
                None => format!("untyped {name}"),
            };
            return Err(RuntimeError::new(format!(
                "incompatible endpoint payload types: cannot connect {} to {}; connected endpoints must have the same T (or all be untyped)",
                describe(if publishing { "Emitter" } else { "Receiver" }, &owner_state.payload_type),
                describe("Channel", &channel_state.payload_type),
            )));
        }
        match (&mut unsafe { owner_cell.get_mut() }.data, handler) {
            (NativeEndpointData::Emitter(channels), None) => {
                if already { return Ok(owner); }
                channels.push(channel);
            }
            (NativeEndpointData::Receiver(receiver), Some(handler)) => {
                if let Some(subscription) = receiver.subscriptions.iter_mut().find(|s| s.channel == channel) {
                    subscription.handler = handler;
                } else {
                    receiver.subscriptions.push(NativeSubscription { channel, handler });
                }
            }
            _ => unreachable!(),
        }
        if !already {
            let NativeEndpointData::Channel { emitters, receivers } = &mut unsafe { channel_cell.get_mut() }.data else { unreachable!() };
            if publishing { emitters.push(owner); } else { receivers.push(owner); }
            self.endpoint_changed(channel);
        }
        self.endpoint_changed(owner);
        Ok(owner)
    }

    fn endpoint_disconnect(&mut self, owner: Value, channel: Value, publishing: bool) -> Result<bool, RuntimeError> {
        let owner_cell = self.endpoint(owner)?;
        let channel_cell = self.endpoint(channel)?;
        if !matches!(unsafe { channel_cell.get() }.data, NativeEndpointData::Channel { .. }) {
            return Err(RuntimeError::new("connection target must be a Channel"));
        }
        let removed = match (&mut unsafe { owner_cell.get_mut() }.data, publishing) {
            (NativeEndpointData::Emitter(channels), true) => {
                let before = channels.len();
                channels.retain(|value| *value != channel);
                channels.len() != before
            }
            (NativeEndpointData::Receiver(receiver), false) => {
                let before = receiver.subscriptions.len();
                receiver.subscriptions.retain(|subscription| subscription.channel != channel);
                receiver.subscriptions.len() != before
            }
            _ => return Err(RuntimeError::new("invalid connection endpoint")),
        };
        let NativeEndpointData::Channel { emitters, receivers } = &mut unsafe { channel_cell.get_mut() }.data else { unreachable!() };
        if publishing { emitters.retain(|value| *value != owner); }
        else { receivers.retain(|value| *value != owner); }
        Ok(removed)
    }

    fn pubsub_emit(&mut self, emitter: Value, data: Value) -> Result<Value, RuntimeError> {
        let cell = self.endpoint(emitter)?;
        let state = unsafe { cell.get() };
        let NativeEndpointData::Emitter(channels) = &state.data else {
            return Err(RuntimeError::new("emit receiver must be an Emitter"));
        };
        if state.closed { return Err(RuntimeError::new("Emitter is closed")); }
        let payload_type = state.payload_type.clone();
        let channels = channels.clone();
        if let Some(expected) = &payload_type { self.heap.bind_contract(data, expected)?; }
        // Snapshot every (receiver, handler) route before queueing anything. The
        // handler is per subscription, so a receiver reached through two channels
        // runs the handler of each channel.
        let mut routes = Vec::new();
        for channel in channels {
            let cell = self.endpoint(channel)?;
            let state = unsafe { cell.get() };
            if state.closed { continue; }
            let NativeEndpointData::Channel { receivers, .. } = &state.data else { unreachable!() };
            for &receiver in receivers {
                let receiver_cell = self.endpoint(receiver)?;
                let receiver_state = unsafe { receiver_cell.get() };
                if receiver_state.closed { continue; }
                let NativeEndpointData::Receiver(subscriptions) = &receiver_state.data else { unreachable!() };
                if let Some(handler) = subscriptions.handler_for(channel) {
                    routes.push((receiver, handler));
                }
            }
        }
        let event = self.allocate_event(data, payload_type);
        for (receiver, handler) in routes {
            let cell = self.endpoint(receiver)?;
            if let NativeEndpointData::Receiver(state) = &mut unsafe { cell.get_mut() }.data {
                state.deliveries += 1;
            }
            // Queued RuntimeEvents are VM roots: receiver, handler and event stay
            // alive until the handler runs, even if the network is dropped.
            self.event_loop.enqueue_microtask(RuntimeEvent::ReceiverDelivery { receiver, callback: handler, event });
        }
        Ok(event)
    }

    fn receiver_handler_function(&self, handler: Value) -> Result<(u32, u8), RuntimeError> {
        let function = match self.heap.get(handler) {
            Some(HeapObject::Closure { function_index, .. }) => *function_index,
            _ => return Err(RuntimeError::new("Receiver handler must be a function")),
        };
        let prepared = &self.prepared.functions[function as usize];
        if prepared.is_concurrent || prepared.arity > 1 {
            return Err(RuntimeError::new("Receiver handler must be a normal/async function taking zero or one Event argument"));
        }
        Ok((function, prepared.arity))
    }

    pub(super) fn deliver_receiver_callback(&mut self, receiver: Value, callback: Value, event: Value) -> Result<Value, RuntimeError> {
        let cell = self.endpoint(receiver)?;
        let state = unsafe { cell.get_mut() };
        if let NativeEndpointData::Receiver(receiver_state) = &mut state.data {
            receiver_state.deliveries = receiver_state.deliveries.saturating_sub(1);
        }
        // Closing a receiver drops deliveries that were queued but not yet run.
        if state.closed { return Ok(Value::NULL); }
        let (function, arity) = self.receiver_handler_function(callback)?;
        let word = event.word();
        let args = if arity == 0 { &[][..] } else { std::slice::from_ref(&word) };
        self.spawn_async_task(function, None, args, Some(callback), None)
    }

    fn pubsub_close(&mut self, endpoint: Value) -> Result<Value, RuntimeError> {
        let cell = self.endpoint(endpoint)?;
        let state = unsafe { cell.get_mut() };
        if state.closed { return Ok(Value::FALSE); }
        let links: Vec<(Value, Value, bool)> = match &state.data {
            NativeEndpointData::Emitter(channels) => channels.iter().map(|&channel| (endpoint, channel, true)).collect(),
            NativeEndpointData::Channel { emitters, receivers } => emitters.iter().map(|&owner| (owner, endpoint, true))
                .chain(receivers.iter().map(|&owner| (owner, endpoint, false))).collect(),
            NativeEndpointData::Receiver(receiver) => receiver.subscriptions.iter()
                .map(|subscription| (endpoint, subscription.channel, false)).collect(),
            NativeEndpointData::Event(_) => return Err(RuntimeError::new("Events cannot be closed")),
        };
        state.closed = true;
        for (owner, channel, publishing) in links { self.endpoint_disconnect(owner, channel, publishing)?; }
        Ok(Value::TRUE)
    }

    fn pubsub_operation(&mut self, native: PubSubNative, receiver: Value, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        use PubSubNative::*;
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        match native {
            EventNew => return Ok(self.allocate_event(arg(0), None)),
            EmitterNew | ChannelNew | ReceiverNew => {
                let data = match native {
                    EmitterNew => NativeEndpointData::Emitter(Vec::new()),
                    ChannelNew => NativeEndpointData::Channel { emitters: Vec::new(), receivers: Vec::new() },
                    _ => NativeEndpointData::Receiver(Box::default()),
                };
                let state = Arc::new(NativeEndpointCell::new(data, None));
                let object = match native {
                    EmitterNew => HeapObject::Emitter { state },
                    ChannelNew => HeapObject::Channel { state },
                    _ => HeapObject::Receiver { state },
                };
                return Ok(self.heap.alloc(object));
            }
            PublishTo => return self.endpoint_connect(receiver, arg(0), None),
            SubscribeFrom => return self.endpoint_connect(receiver, arg(0), Some(arg(1))),
            UnpublishFrom | UnsubscribeFrom => return self.endpoint_disconnect(receiver, arg(0), native == UnpublishFrom).map(Value::from_bool),
            Emit => return self.pubsub_emit(receiver, arg(0)),
            Close => return self.pubsub_close(receiver),
            _ => {}
        }
        let cell = self.endpoint(receiver)?;
        let state = unsafe { cell.get() };
        if native == Closed { return Ok(Value::from_bool(state.closed)); }
        let count = match (native, &state.data) {
            (EventData, NativeEndpointData::Event(value)) => return Ok(*value),
            (ChannelCount, NativeEndpointData::Emitter(channels)) => channels.len(),
            (ChannelCount, NativeEndpointData::Receiver(state)) => state.subscriptions.len(),
            (EmitterCount, NativeEndpointData::Channel { emitters, .. }) => emitters.len(),
            (ReceiverCount, NativeEndpointData::Channel { receivers, .. }) => receivers.len(),
            _ => return Err(RuntimeError::new("invalid pub/sub operation")),
        };
        Ok(Value::from_num(count as f64))
    }

    pub(super) fn execute_pubsub_native(&mut self, op: &PreparedOp, native: PubSubNative, receiver: RegisterWord, args: &[RegisterWord]) {
        self.mark_gc_point(op);
        self.maybe_collect();
        match self.pubsub_operation(native, Value::from_word(receiver), args) {
            Ok(value) => unsafe { self.set_reg(op.a, value.word()); self.advance(op); },
            Err(error) => self.fail_at(op, error.message),
        }
    }
}
