// Event / Emitter / Channel / Receiver: independently created endpoints wired
// many-to-many. Emitters publish to channels; receivers subscribe from channels
// with one handler per channel. Every emit() wraps its payload in one Event that
// is handed to the handler of each (receiver, channel) subscription it reaches.

fn pubsub_runtime_error(source: &str) -> String {
    let (program, entry, manifest) = compile_typescript_module(source);
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).expect_err("expected a runtime pub/sub error").message
}

#[test]
fn napalm_pubsub_routes_events_many_to_many_through_per_channel_handlers() {
    run_napalm(r#"
        const news = new Channel();
        const sports = new Channel();
        const a = new Emitter();
        const b = new Emitter();
        a.publish_to(news).publish_to(sports);
        b.publishTo(sports);
        const r1 = new Receiver();
        const r2 = new Receiver();
        const result = [];
        r1.subscribe_from(news, (event) => { result.push(`r1/news:${event.data.n}`); });
        r2.subscribe_from(sports, (event) => { result.push(`r2/sports:${event.data.n}`); })
          .subscribeFrom(news, (event) => { result.push(`r2/news:${event.data.n}`); });
        a.emit({ n: 1 });
        b.emit({ n: 2 });
        // Handlers are microtasks: these synchronous pushes land first.
        result.push(news.emitterCount);
        result.push(sports.emitterCount);
        result.push(news.receiverCount);
        result.push(sports.receiverCount);
        result.push(a.channelCount);
        result.push(r2.channelCount);
    "#, "[1, 2, 2, 1, 2, 2, r1/news:1, r2/news:1, r2/sports:1, r2/sports:2]");
}

#[test]
fn napalm_pubsub_handler_receives_the_emitted_event() {
    run_napalm(r#"
        const channel = new Channel<string>();
        const emitter = new Emitter<string>();
        const receiver = new Receiver<string>();
        emitter.publish_to(channel);
        const silent = new Receiver<string>();
        let sent: any = null;
        const result = [];
        receiver.subscribe_from(channel, (event) => { result.push("first handler"); });
        // Subscribing again to the same channel replaces that channel's handler.
        receiver.subscribe_from(channel, (event) => {
            result.push(event == sent);
            result.push(event.data);
        });
        silent.subscribe_from(channel, () => { result.push("no-arg handler"); });
        sent = emitter.emit("hello");
        result.push(channel.receiverCount);
    "#, "[2, true, hello, no-arg handler]");
}

#[test]
fn napalm_pubsub_unsubscribe_and_unpublish_cut_routes() {
    run_napalm(r#"
        const channel = new Channel();
        const emitter = new Emitter();
        const receiver = new Receiver();
        emitter.publish_to(channel);
        const result = [];
        receiver.subscribe_from(channel, (event) => { result.push(event.data); });
        emitter.emit(1); // already queued: still delivered after unsubscribing
        result.push(receiver.unsubscribe_from(channel));
        result.push(receiver.unsubscribeFrom(channel));
        emitter.emit(2);
        receiver.subscribe_from(channel, (event) => { result.push(event.data); });
        emitter.emit(3);
        result.push(emitter.unpublish_from(channel));
        result.push(channel.emitterCount);
        emitter.emit(4);
    "#, "[true, false, true, 0, 1, 3]");
}

#[test]
fn napalm_pubsub_close_disconnects_and_drops_pending_deliveries() {
    run_napalm(r#"
        const channel = new Channel();
        const emitter = new Emitter();
        const receiver = new Receiver();
        const other = new Receiver();
        emitter.publish_to(channel);
        const result = [];
        receiver.subscribe_from(channel, (event) => { result.push(`receiver:${event.data}`); });
        other.subscribe_from(channel, (event) => { result.push(`other:${event.data}`); });
        emitter.emit(1);
        result.push(receiver.close());
        result.push(receiver.close());
        result.push(receiver.closed);
        result.push(channel.receiverCount);
        channel.close();
        result.push(other.channelCount);
        result.push(emitter.channelCount);
        emitter.emit(2);
    "#, "[true, false, true, 1, 0, 0, other:1]");
}

#[test]
fn napalm_pubsub_async_handlers_can_await() {
    run_napalm(r#"
        const channel = new Channel<number>();
        const emitter = new Emitter<number>();
        const receiver = new Receiver<number>();
        emitter.publish_to(channel);
        const result = [];
        receiver.subscribe_from(channel, async function (event) {
            const doubled = await Promise.resolve(event.data * 2);
            result.push(doubled);
        });
        emitter.emit(21);
    "#, "[42]");
}

#[test]
fn napalm_pubsub_event_carries_payload_and_typeof_reports_generics() {
    run_napalm(r#"
        const standalone = new Event<string>("x");
        const empty = new Event();
        const emitter = new Emitter<number>();
        const untyped = new Emitter();
        const result = [
            standalone.data, empty.data, untyped.emit().data, typeof standalone, typeof emitter.emit(1),
            typeof new Receiver<string>(), typeof new Channel(),
        ];
    "#, "[x, null, null, Event<String>, Event<Number>, Receiver<String>, Channel]");
}

#[test]
fn napalm_pubsub_rejects_statically_invalid_connections() {
    let messages = contract_diagnostics(r#"
        interface Noded { id: number; }
        class Bird { name: string = "tweety"; }
        const emitter = new Emitter<Noded>();
        const channel = new Channel<Bird>();
        emitter.publish_to(channel);
        const receiver = new Receiver<number>();
        receiver.subscribe_from(new Channel<string>(), (event) => {});
        receiver.subscribe_from(new Emitter<number>(), (event) => {});
        receiver.subscribe_from(new Channel<number>());
    "#);
    assert!(
        messages.iter().any(|message| message.contains("cannot connect Emitter<Noded> to Channel<Bird>")),
        "{messages:?}"
    );
    assert_eq!(
        messages.iter().filter(|message| message.contains("payload types must match")).count(), 2,
        "{messages:?}"
    );
    assert!(messages.iter().any(|message| message.contains("expects a Channel")), "{messages:?}");
    assert!(messages.iter().any(|message| message.contains("requires a handler")), "{messages:?}");
}

#[test]
fn napalm_pubsub_rejects_invalid_connections_at_runtime() {
    // Typed endpoints keep their contract behind `any`.
    let error = pubsub_runtime_error(r#"
        const emitter = new Emitter<number>();
        const channel: any = new Channel<string>();
        emitter.publish_to(channel);
        let result = 1;
    "#);
    assert!(error.contains("cannot connect Emitter<Number> to Channel<String>"), "{error}");

    // A typed endpoint cannot connect to an untyped one either.
    let error = pubsub_runtime_error(r#"
        const receiver = new Receiver<number>();
        receiver.subscribe_from(new Channel(), (event) => {});
        let result = 1;
    "#);
    assert!(error.contains("cannot connect Receiver<Number> to untyped Channel"), "{error}");

    let error = pubsub_runtime_error(r#"
        const channel = new Channel<number>();
        const emitter = new Emitter<number>();
        emitter.publish_to(channel);
        const dynamicChannel: any = channel;
        const narrowed: Channel<string> = dynamicChannel;
        let result = 1;
    "#);
    assert!(error.contains("incompatible payload contract"), "{error}");

    let error = pubsub_runtime_error(r#"
        const receiver: any = new Receiver();
        receiver.subscribe_from(new Channel());
        let result = 1;
    "#);
    assert!(error.contains("requires a handler"), "{error}");

    let error = pubsub_runtime_error(r#"
        const receiver: any = new Receiver();
        receiver.subscribe_from(new Channel(), 5);
        let result = 1;
    "#);
    assert!(error.contains("handler must be a function"), "{error}");

    let error = pubsub_runtime_error(r#"
        const emitter = new Emitter();
        emitter.close();
        emitter.emit(1);
        let result = 1;
    "#);
    assert!(error.contains("Emitter is closed"), "{error}");
}

#[test]
fn napalm_pubsub_typed_emitter_validates_payload() {
    let error = pubsub_runtime_error(r#"
        const emitter = new Emitter<number>();
        const dynamicEmitter: any = emitter;
        dynamicEmitter.emit("not a number");
        let result = 1;
    "#);
    assert!(error.contains("Num"), "{error}");

    // null does not satisfy T, so a typed Emitter needs a payload.
    let error = pubsub_runtime_error(r#"
        const emitter = new Emitter<number>();
        emitter.emit();
        let result = 1;
    "#);
    assert!(error.contains("Num"), "{error}");
}
