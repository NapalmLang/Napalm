#[test]
fn napalm_http_client_helpers_can_be_shadowed() {
    run_napalm(
        r#"
        function get() { return "user get"; }
        const getJSON = () => "user getJSON";
        let postJSON = () => "user postJSON";
        function nested(get) { return () => get(); }
        const captured = nested(() => "local get");
        const server = listen "127.0.0.1:0" {
            route "GET" ["/"] (req) => get();
            route "GET" ["/json"] (req) => ({body: "{\"answer\":42}"});
            route "GET" ["/header"] (req) => req.headers["x-test"];
            route "POST" ["/echo"] (req) => ({body: req.body});
        };
        const watchdog = setTimeout(() => http.close(server), 5000);
        const base = "http://" + http.address(server);
        const a = await http.get(base + "/");
        const b = await http.get(base + "/header", [["X-Test", "pairs"]]);
        const c = await http.getJSON(base + "/json");
        const d = await http.getJSON(base + "/json", {});
        const e = await http.postJSON(base + "/echo", "{\"answer\":7}");
        const f = await http.postJSON(base + "/echo", "{\"answer\":8}", {});
        const result = [get(), getJSON(), postJSON(), captured(), a.body, b.body, c.json.answer, d.json.answer, e.json.answer, f.json.answer];
        http.close(server); watchdog.cancel();
        "#,
        "[user get, user getJSON, user postJSON, local get, user get, pairs, 42, 42, 7, 8]",
    );
}

#[test]
fn napalm_http_client_helper_shadowing_is_lexical() {
    run_napalm(
        r#"
        const nativeGet = get;
        const nativeGetJSON = getJSON;
        const nativePostJSON = postJSON;
        let local = null;
        {
            function get() { return 1; }
            function getJSON() { return 2; }
            function postJSON() { return 3; }
            local = () => [get(), getJSON(), postJSON()];
        }
        System.gc();
        const server = listen "127.0.0.1:0" {
            route "GET" ["/"] (req) => "42";
            route "POST" ["/"] (req) => req.body;
        };
        const watchdog = setTimeout(() => http.close(server), 5000);
        const url = "http://" + http.address(server) + "/";
        const a = await get(url);
        const b = await getJSON(url);
        const c = await postJSON(url, "7");
        const d = await nativeGet(url);
        const e = await nativeGetJSON(url);
        const f = await nativePostJSON(url, "8");
        const result = [local(), a.body, b.json, c.json, d.body, e.json, f.json];
        http.close(server); watchdog.cancel();
        "#,
        "[[1, 2, 3], 42, 42, 7, 42, 42, 8]",
    );
}

#[test]
fn napalm_http_routes_async_clients_and_gc() {
    run_napalm(
        r#"
        function serve(prefix) {
            return listen "127.0.0.1:0" {
                route "GET" ["/", "/hello"] (req) => ({status: 201, body: prefix, headers: [["X-Test", "yes"]]});
                route "GET" ["/user/:id"] async (req) => {
                    await Timer.delay(1);
                    return {body: req.params.id + ":" + req.query};
                };
                route "GET" ["/files/*"] (req) => req.params.wildcard;
                route "POST" ["/echo"] (req) => ({body: req.body, headers: {"Content-Type": req.headers["content-type"]}});
                route "GET" ["/json"] (req) => ({body: "{\"answer\":42}"});
            };
        }
        const server = serve("hello ñ");
        const watchdog = setTimeout(() => http.close(server), 5000);
        System.gc();
        const base = "http://" + http.address(server);
        const a = await get(base + "/hello?ignored=1");
        const b = await get(base + "/user/7?q=x");
        const c = await get(base + "/files/a/b");
        const d = await postJSON(base + "/echo", "{\"a\":1}", [["X-Foo", "Bar"]]);
        const e = await getJSON(base + "/json");
        const missing = await get(base + "/missing");
        const wrong = await get(base + "/echo");
        const result = [a.status, a.body, a.headers["x-test"], b.body, c.body, d.json.a, d.headers["content-type"], e.json.answer, missing.status, wrong.status, wrong.headers.allow, http.close(server), http.close(server)];
        watchdog.cancel();
    "#,
        "[201, hello ñ, yes, 7:q=x, a/b, 1, application/json, 42, 404, 405, POST, true, false]",
    );
}

#[test]
fn napalm_http_stream_timers_and_handler_failures() {
    run_napalm(
        r#"
        const server = listen "127.0.0.1:0" {
            route "GET" ["/stream"] (req) => {
                http.startStream(req, 200, [["Content-Type", "text/event-stream"]]);
                setTimeout(() => http.streamWrite(req.serverId, req.reqId, "data: ñ\n\n"), 0);
                setTimeout(() => { http.streamWrite(req.serverId, req.reqId, "data: 2\n\n"); http.streamEnd(req.serverId, req.reqId); }, 10);
                return 0;
            };
            route "GET" ["/throw"] (req) => { throw "private failure"; };
            route "GET" ["/async-throw"] async (req) => { await Timer.delay(0); throw "private failure"; };
            route "GET" ["/bad"] (req) => 0;
            route "GET" ["/header"] (req) => ({body: "bad", headers: [["X-Test", "bad\r\ninjected"]]});
            route "GET" ["/framing"] (req) => ({body: "bad", headers: {"Content-Length": "999"}});
            route "GET" ["/ok"] (req) => "alive";
        };
        const watchdog = setTimeout(() => http.close(server), 5000);
        const base = "http://" + http.address(server);
        const stream = await get(base + "/stream");
        const a = await get(base + "/throw");
        const b = await get(base + "/async-throw");
        const c = await get(base + "/bad");
        const d = await get(base + "/header");
        const e = await get(base + "/framing");
        const f = await get(base + "/ok");
        let invalidJson = false;
        try { await getJSON(base + "/ok"); } catch (error) { invalidJson = true; }
        const result = [stream.status, stream.body == "data: ñ\n\ndata: 2\n\n", stream.headers["content-type"], a.status, a.body, b.status, c.status, d.status, e.status, f.body, invalidJson];
        http.close(server); watchdog.cancel();
    "#,
        "[200, true, text/event-stream, 500, Internal Server Error, 500, 500, 500, 500, alive, true]",
    );
}

#[test]
fn napalm_http_validation_and_independent_servers() {
    run_napalm(
        r#"
        let errors = 0;
        try { listen "127.0.0.1:0" { route "GET" ["/", "/"] (req) => "duplicate"; } } catch (e) { errors += 1; }
        try { listen "127.0.0.1:0" { route "GET" ["bad"] (req) => "bad"; } } catch (e) { errors += 1; }
        try { listen "127.0.0.1:0" { route "GET" ["/"] () => "bad"; } } catch (e) { errors += 1; }
        const a = listen "127.0.0.1:0" { route "GET" ["/"] (req) => "one"; };
        const b = listen "127.0.0.1:0" { route "GET" ["/"] (req) => "two"; };
        const wa = setTimeout(() => http.close(a), 5000);
        const wb = setTimeout(() => http.close(b), 5000);
        const address = http.address(a);
        try { listen address { route "GET" ["/"] (req) => "busy"; } } catch (e) { errors += 1; }
        const responses = await Promise.all([get("http://" + address + "/"), get("http://" + http.address(b) + "/")]);
        const result = [errors, responses[0].body, responses[1].body];
        http.close(a); http.close(b); wa.cancel(); wb.cancel();
    "#,
        "[4, one, two]",
    );
}

#[test]
fn napalm_http_parser_rejects_malformed_routes() {
    for source in [
        "listen '127.0.0.1:0' { route 'GET' [] (req) => 'bad'; }",
        "listen '127.0.0.1:0' { rout 'GET' ['/'] (req) => 'bad'; }",
        "listen '127.0.0.1:0' { route 'GET' ['/'] (req) => 'bad';",
    ] {
        assert!(
            crate::napalm_front_main::parse("bad.nap", source).has_errors(),
            "accepted {source}"
        );
    }
}

#[test]
fn napalm_http_contextual_listen_and_fetch_header_pairs() {
    run_napalm(
        r#"
        let listen = 7;
        listen = listen + 1;
        const server = http.listen("127.0.0.1:0", [["GET", ["/"], (req) => req.headers["x-test"]]]);
        const watchdog = setTimeout(() => http.close(server), 5000);
        const response = await fetch("http://" + http.address(server) + "/", {headers: [["X-Test", "pairs"]]});
        const result = [listen, response.status, await response.text()];
        http.close(server); watchdog.cancel();
    "#,
        "[8, 200, pairs]",
    );
}
