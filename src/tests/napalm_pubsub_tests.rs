// Event / Emitter / Channel / Receiver: independently created endpoints wired
// many-to-many. Emitters publish to channels, receivers subscribe from channels,
// and every emit() wraps its payload in one Event routed to all reachable receivers.

fn pubsub_runtime_error(source: &str) -> String {
    let (program, entry, manifest) = compile_typescript_module(source);
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).expect_err("expected a runtime pub/sub error").message
}

#[test]
fn napalm_pubsub_routes_events_many_to_many_once_per_receiver() {
    run_napalm(r#"
        const news = new Channel();
        const sports = new Channel();
        const a = new Emitter();
        const b = new Emitter();
        a.publish_to(news).publish_to(sports);
        b.publishTo(sports);
        const r1 = new Receiver();
        const r2 = new Receiver();
        r1.subscribe_from(news);
        r2.subscribe_from(sports).subscribeFrom(news);
        const sent = a.emit({ n: 1 });
        b.emit({ n: 2 });
        const e1 = r1.tryReceive();
        const e2a = r2.tryReceive();
        const e2b = r2.try_receive();
        const result = [
            e1.data.n, e2a.data.n, e2b.data.n, r1.tryReceive(), r2.tryReceive(),
            e1 == sent, e2a == sent, news.emitterCount, sports.emitterCount,
            sports.receiverCount, a.channelCount, r2.channelCount,
        ];
    "#, "[1, 1, 2, null, null, true, true, 1, 2, 1, 2, 2]");
}

#[test]
fn napalm_pubsub_unsubscribe_and_unpublish_cut_routes() {
    run_napalm(r#"
        const channel = new Channel();
        const emitter = new Emitter();
        const receiver = new Receiver();
        emitter.publish_to(channel);
        receiver.subscribe_from(channel);
        emitter.emit(1);
        const removed = receiver.unsubscribe_from(channel);
        const again = receiver.unsubscribeFrom(channel);
        emitter.emit(2);
        receiver.subscribe_from(channel);
        const unpublished = emitter.unpublish_from(channel);
        emitter.emit(3);
        const result = [receiver.length, receiver.tryReceive().data, receiver.length, removed, again, unpublished, channel.emitterCount];
    "#, "[1, 1, 0, true, false, true, 0]");
}

#[test]
fn napalm_pubsub_receive_waits_for_events_and_close_ends_with_null() {
    run_napalm(r#"
        class Probe {
            static async run() {
                const channel = new Channel<number>();
                const emitter = new Emitter<number>();
                const receiver = new Receiver<number>();
                emitter.publish_to(channel);
                receiver.subscribe_from(channel);
                const pending = receiver.receive();
                emitter.emit(10);
                emitter.emit(20);
                const first = await pending;
                const second = await receiver.receive();
                receiver.close();
                const end = await receiver.receive();
                return [first.data, second.data, end, receiver.closed, channel.receiverCount];
            }
        }
        const result = await Probe.run();
    "#, "[10, 20, null, true, 0]");
}

#[test]
fn napalm_pubsub_receiver_callbacks_support_on_once_and_off() {
    run_napalm(r#"
        const channel = new Channel<string>();
        const emitter = new Emitter<string>();
        const receiver = new Receiver<string>();
        emitter.publish_to(channel);
        receiver.subscribe_from(channel);
        let token: any = null;
        const result = [];
        token = receiver.on((event) => { result.push("on:" + event.data); });
        receiver.once((event) => { result.push("once:" + event.data); });
        emitter.emit("a");
        emitter.emit("b");
        receiver.off(token);
        emitter.emit("c");
        // Callbacks are microtasks; with no listener left, "c" is queued instead.
        result.push(receiver.length);
    "#, "[1, on:a, once:a, on:b]");
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
fn napalm_pubsub_rejects_statically_mismatched_generic_connections() {
    let messages = contract_diagnostics(r#"
        interface Noded { id: number; }
        class Bird { name: string = "tweety"; }
        const emitter = new Emitter<Noded>();
        const channel = new Channel<Bird>();
        emitter.publish_to(channel);
        const receiver = new Receiver<number>();
        receiver.subscribe_from(new Channel<string>());
        receiver.subscribe_from(new Emitter<number>());
    "#);
    assert_eq!(
        messages.iter().filter(|message| message.contains("payload types must match")).count(), 2,
        "{messages:?}"
    );
    assert!(messages.iter().any(|message| message.contains("expects a Channel")), "{messages:?}");
}

#[test]
fn napalm_pubsub_rejects_mismatched_generic_connections_at_runtime() {
    // Typed endpoints keep their contract behind `any`.
    let error = pubsub_runtime_error(r#"
        const emitter = new Emitter<number>();
        const channel: any = new Channel<string>();
        emitter.publish_to(channel);
        let result = 1;
    "#);
    assert!(error.contains("incompatible endpoint payload types"), "{error}");

    // A typed endpoint cannot connect to an untyped one either.
    let error = pubsub_runtime_error(r#"
        const receiver = new Receiver<number>();
        receiver.subscribe_from(new Channel());
        let result = 1;
    "#);
    assert!(error.contains("incompatible endpoint payload types"), "{error}");

    let error = pubsub_runtime_error(r#"
        const channel = new Channel<number>();
        const emitter = new Emitter<number>();
        emitter.publish_to(channel);
        const dynamicChannel: any = channel;
        const narrowed: Channel<string> = dynamicChannel;
        let result = 1;
    "#);
    assert!(error.contains("incompatible payload contract"), "{error}");

    // A typed Emitter rejects a payload-less emit(): null does not satisfy T.
    let error = pubsub_runtime_error(r#"
        const emitter = new Emitter<number>();
        emitter.emit();
        let result = 1;
    "#);
    assert!(error.contains("Num"), "{error}");
}

#[test]
fn napalm_pubsub_typed_emitter_validates_payload_behind_any_alias() {
    let error = pubsub_runtime_error(r#"
        const emitter = new Emitter<number>();
        const dynamicEmitter: any = emitter;
        dynamicEmitter.emit("not a number");
        let result = 1;
    "#);
    assert!(error.contains("Num"), "{error}");
}
