//! VM-local many-to-many event routing. Connections are GC edges, not Arc
//! cycles. Each receiver owns its queue, waiters and callback registrations.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PubSubNative {
    EventNew, EventData, EmitterNew, ChannelNew, ReceiverNew,
    PublishTo, UnpublishFrom, SubscribeFrom, UnsubscribeFrom, Emit,
    Receive, TryReceive, On, Once, Off, ListenerCount, Length,
    ChannelCount, EmitterCount, ReceiverCount, Close, Closed,
}

#[derive(Debug, Default)]
pub(super) struct NativeReceiverState {
    channels: Vec<Value>,
    queue: VecDeque<Value>,
    waiters: VecDeque<Value>,
    listeners: Vec<NativeEventListener>,
    next_token: u64,
    deliveries: usize,
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
                receiver.channels.iter().chain(&receiver.queue).chain(&receiver.waiters)
                    .copied().for_each(&mut visit);
                receiver.listeners.iter().for_each(|listener| visit(listener.callback));
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
                + (receiver.channels.capacity() + receiver.queue.capacity() + receiver.waiters.capacity())
                    * std::mem::size_of::<Value>()
                + receiver.listeners.capacity() * std::mem::size_of::<NativeEventListener>(),
        }
    }

    // Retyping a live network would silently invalidate its other endpoints or
    // promises. Install an endpoint's generic contract before connecting/using it.
    pub(super) fn can_set_payload_type(&self) -> bool {
        match &self.data {
            NativeEndpointData::Event(_) => true,
            NativeEndpointData::Emitter(channels) => channels.is_empty(),
            NativeEndpointData::Channel { emitters, receivers } => emitters.is_empty() && receivers.is_empty(),
            NativeEndpointData::Receiver(receiver) => receiver.channels.is_empty()
                && receiver.queue.is_empty() && receiver.waiters.is_empty() && receiver.deliveries == 0,
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

    fn endpoint_connect(&mut self, owner: Value, channel: Value, publishing: bool) -> Result<Value, RuntimeError> {
        let owner_cell = self.endpoint(owner)?;
        let channel_cell = self.endpoint(channel)?;
        let owner_state = unsafe { owner_cell.get() };
        let channel_state = unsafe { channel_cell.get() };
        if !matches!(channel_state.data, NativeEndpointData::Channel { .. }) {
            return Err(RuntimeError::new("connection target must be a Channel"));
        }
        let channels = match (&owner_state.data, publishing) {
            (NativeEndpointData::Emitter(channels), true) => channels,
            (NativeEndpointData::Receiver(receiver), false) => &receiver.channels,
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
        if channels.contains(&channel) { return Ok(owner); }
        match &mut unsafe { owner_cell.get_mut() }.data {
            NativeEndpointData::Emitter(channels) => channels.push(channel),
            NativeEndpointData::Receiver(receiver) => receiver.channels.push(channel),
            _ => unreachable!(),
        }
        let NativeEndpointData::Channel { emitters, receivers } = &mut unsafe { channel_cell.get_mut() }.data else { unreachable!() };
        if publishing { emitters.push(owner); } else { receivers.push(owner); }
        self.endpoint_changed(owner);
        self.endpoint_changed(channel);
        Ok(owner)
    }

    fn endpoint_disconnect(&mut self, owner: Value, channel: Value, publishing: bool) -> Result<bool, RuntimeError> {
        let owner_cell = self.endpoint(owner)?;
        let channel_cell = self.endpoint(channel)?;
        if !matches!(unsafe { channel_cell.get() }.data, NativeEndpointData::Channel { .. }) {
            return Err(RuntimeError::new("connection target must be a Channel"));
        }
        let channels = match (&mut unsafe { owner_cell.get_mut() }.data, publishing) {
            (NativeEndpointData::Emitter(channels), true) => channels,
            (NativeEndpointData::Receiver(receiver), false) => &mut receiver.channels,
            _ => return Err(RuntimeError::new("invalid connection endpoint")),
        };
        let before = channels.len();
        channels.retain(|value| *value != channel);
        let removed = channels.len() != before;
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
        // Snapshot every route first. One emit is one Event: a receiver reachable
        // through several channels still observes it exactly once, in first-route order.
        let mut destinations = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for channel in channels {
            let cell = self.endpoint(channel)?;
            let state = unsafe { cell.get() };
            if state.closed { continue; }
            let NativeEndpointData::Channel { receivers, .. } = &state.data else { unreachable!() };
            destinations.extend(receivers.iter().copied().filter(|receiver| seen.insert(receiver.word().0)));
        }
        let event = self.allocate_event(data, payload_type);
        let root = self.add_root(event);
        let roots: Vec<_> = destinations.iter().map(|&value| self.add_root(value)).collect();
        let delivered = (|| {
            for receiver in destinations { self.pubsub_deliver(receiver, event)?; }
            Ok(event)
        })();
        for root in roots { self.release_root(root); }
        self.release_root(root);
        delivered
    }

    fn pubsub_deliver(&mut self, receiver: Value, event: Value) -> Result<(), RuntimeError> {
        let cell = self.endpoint(receiver)?;
        if unsafe { cell.get() }.closed { return Ok(()); }
        let NativeEndpointData::Receiver(state) = &mut unsafe { cell.get_mut() }.data else { unreachable!() };
        // Cancelled receives must not consume an event that a later waiter needs.
        let mut waiter = None;
        while let Some(promise) = state.waiters.pop_front() {
            if self.promise_cell(promise).is_ok_and(|cell|
                matches!(unsafe { cell.get() }, PromiseState::Pending { .. })) {
                waiter = Some(promise);
                break;
            }
        }
        let callbacks: Vec<_> = state.listeners.iter().map(|listener| listener.callback).collect();
        state.listeners.retain(|listener| !listener.once);
        state.deliveries += callbacks.len();
        // Callback mode doesn't accumulate an unread queue. A pending receive
        // still gets its event, independently of registered callback observers.
        if waiter.is_none() && callbacks.is_empty() { state.queue.push_back(event); }
        self.endpoint_changed(receiver);
        for callback in callbacks {
            self.event_loop.enqueue_microtask(RuntimeEvent::ReceiverDelivery { receiver, callback, event });
        }
        if let Some(promise) = waiter {
            let root = self.add_root(promise);
            let result = self.settle_promise_outcome(promise, PromiseOutcome::Fulfilled(event));
            self.release_root(root);
            result?;
        }
        Ok(())
    }

    fn pubsub_receive(&mut self, receiver: Value, wait: bool) -> Result<Value, RuntimeError> {
        let cell = self.endpoint(receiver)?;
        let state = unsafe { cell.get_mut() };
        let NativeEndpointData::Receiver(receiver_state) = &mut state.data else {
            return Err(RuntimeError::new("receive receiver must be a Receiver"));
        };
        let immediate = receiver_state.queue.pop_front().or_else(|| state.closed.then_some(Value::NULL));
        if !wait { return Ok(immediate.unwrap_or(Value::NULL)); }
        if let Some(value) = immediate {
            let root = self.add_root(value);
            let promise = self.allocate_promise(PromiseState::Fulfilled(value));
            self.release_root(root);
            return Ok(promise);
        }
        let promise = self.allocate_pending_promise();
        receiver_state.waiters.push_back(promise);
        self.endpoint_changed(receiver);
        Ok(promise)
    }

    fn receiver_callback_function(&self, callback: Value) -> Result<(u32, u8), RuntimeError> {
        let function = match self.heap.get(callback) {
            Some(HeapObject::Closure { function_index, .. }) => *function_index,
            _ => return Err(RuntimeError::new("Receiver listener must be a function")),
        };
        let prepared = &self.prepared.functions[function as usize];
        if prepared.is_concurrent || prepared.arity > 1 {
            return Err(RuntimeError::new("Receiver listener must be a normal/async function taking zero or one Event argument"));
        }
        Ok((function, prepared.arity))
    }

    fn pubsub_listen(&mut self, receiver: Value, callback: Value, once: bool) -> Result<Value, RuntimeError> {
        self.receiver_callback_function(callback)?;
        let cell = self.endpoint(receiver)?;
        let state = unsafe { cell.get_mut() };
        if state.closed { return Err(RuntimeError::new("Receiver is closed")); }
        let NativeEndpointData::Receiver(state) = &mut state.data else {
            return Err(RuntimeError::new("listener receiver must be a Receiver"));
        };
        let token = state.next_token + 1;
        if token > 9_007_199_254_740_991 { return Err(RuntimeError::new("Receiver listener tokens exhausted")); }
        state.next_token = token;
        state.listeners.push(NativeEventListener { token, callback, once });
        self.endpoint_changed(receiver);
        Ok(Value::from_num(token as f64))
    }

    fn pubsub_off(&mut self, receiver: Value, key: Value) -> Result<Value, RuntimeError> {
        let token = if let Some(number) = key.as_num() {
            if !number.is_finite() || number < 1.0 || number.fract() != 0.0 || number > 9_007_199_254_740_991.0 {
                return Err(RuntimeError::new("Receiver.off expects a listener token or function"));
            }
            Some(number as u64)
        } else if matches!(self.heap.get(key), Some(HeapObject::Closure { .. })) { None }
        else { return Err(RuntimeError::new("Receiver.off expects a listener token or function")); };
        let cell = self.endpoint(receiver)?;
        let NativeEndpointData::Receiver(state) = &mut unsafe { cell.get_mut() }.data else {
            return Err(RuntimeError::new("off receiver must be a Receiver"));
        };
        let before = state.listeners.len();
        state.listeners.retain(|listener| token.map_or(listener.callback != key, |token| listener.token != token));
        Ok(Value::from_bool(state.listeners.len() != before))
    }

    pub(super) fn deliver_receiver_callback(&mut self, receiver: Value, callback: Value, event: Value) -> Result<Value, RuntimeError> {
        let cell = self.endpoint(receiver)?;
        if let NativeEndpointData::Receiver(state) = &mut unsafe { cell.get_mut() }.data {
            state.deliveries = state.deliveries.saturating_sub(1);
        }
        let (function, arity) = self.receiver_callback_function(callback)?;
        let word = event.word();
        let args = if arity == 0 { &[][..] } else { std::slice::from_ref(&word) };
        self.spawn_async_task(function, None, args, Some(callback), None)
    }

    fn pubsub_close(&mut self, endpoint: Value) -> Result<Value, RuntimeError> {
        let cell = self.endpoint(endpoint)?;
        let state = unsafe { cell.get_mut() };
        if state.closed { return Ok(Value::FALSE); }
        state.closed = true;
        let mut waiters = Vec::new();
        let mut links = Vec::new();
        match &mut state.data {
            NativeEndpointData::Emitter(channels) => {
                links.extend(channels.iter().map(|&channel| (endpoint, channel, true)));
            }
            NativeEndpointData::Channel { emitters, receivers } => {
                links.extend(emitters.iter().map(|&owner| (owner, endpoint, true)));
                links.extend(receivers.iter().map(|&owner| (owner, endpoint, false)));
            }
            NativeEndpointData::Receiver(state) => {
                links.extend(state.channels.iter().map(|&channel| (endpoint, channel, false)));
                state.listeners.clear();
                waiters.extend(state.waiters.drain(..));
            }
            NativeEndpointData::Event(_) => return Err(RuntimeError::new("Events cannot be closed")),
        }
        for (owner, channel, publishing) in links { self.endpoint_disconnect(owner, channel, publishing)?; }
        let roots: Vec<_> = waiters.iter().map(|&promise| self.add_root(promise)).collect();
        let result = (|| {
            for promise in waiters { self.settle_promise_outcome(promise, PromiseOutcome::Fulfilled(Value::NULL))?; }
            Ok(Value::TRUE)
        })();
        for root in roots { self.release_root(root); }
        result
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
            PublishTo | SubscribeFrom => return self.endpoint_connect(receiver, arg(0), native == PublishTo),
            UnpublishFrom | UnsubscribeFrom => return self.endpoint_disconnect(receiver, arg(0), native == UnpublishFrom).map(Value::from_bool),
            Emit => return self.pubsub_emit(receiver, arg(0)),
            Receive | TryReceive => return self.pubsub_receive(receiver, native == Receive),
            On | Once => return self.pubsub_listen(receiver, arg(0), native == Once),
            Off => return self.pubsub_off(receiver, arg(0)),
            Close => return self.pubsub_close(receiver),
            _ => {}
        }
        let cell = self.endpoint(receiver)?;
        let state = unsafe { cell.get() };
        if native == Closed { return Ok(Value::from_bool(state.closed)); }
        let count = match (native, &state.data) {
            (EventData, NativeEndpointData::Event(value)) => return Ok(*value),
            (ChannelCount, NativeEndpointData::Emitter(channels)) => channels.len(),
            (ChannelCount, NativeEndpointData::Receiver(state)) => state.channels.len(),
            (EmitterCount, NativeEndpointData::Channel { emitters, .. }) => emitters.len(),
            (ReceiverCount, NativeEndpointData::Channel { receivers, .. }) => receivers.len(),
            (ListenerCount, NativeEndpointData::Receiver(state)) => state.listeners.len(),
            (Length, NativeEndpointData::Receiver(state)) => state.queue.len(),
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
