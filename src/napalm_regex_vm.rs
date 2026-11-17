//! Bindings to regexr's public string API. Native values own their input and
//! contain no GC references; returned matches survive their regex and source.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RegexNative {
    New,
    BuilderNew,
    Escape,
    AsStr,
    EngineName,
    CaptureNames,
    IsMatch,
    TryIsMatch,
    Find,
    TryFind,
    FindIter,
    Captures,
    TryCaptures,
    CapturesIter,
    Replace,
    ReplaceAll,
    Jit,
    OptimizePrefixes,
    BacktrackLimit,
    SizeLimit,
    NestLimit,
    Build,
    Start,
    End,
    AsBytes,
    Range,
    Len,
    IsEmpty,
    Get,
    Name,
    Index,
}

type Span = (usize, usize);

#[derive(Clone, Debug)]
pub(super) enum RegexObject {
    Compiled(Arc<regexr::Regex>),
    Builder {
        builder: regexr::RegexBuilder,
        pattern_len: usize,
    },
    Match {
        text: Arc<str>,
        span: Span,
    },
    Captures {
        text: Arc<str>,
        slots: Arc<[Option<Span>]>,
        names: Arc<HashMap<String, Option<Span>>>,
    },
}

impl RegexObject {
    pub(super) fn class(&self) -> BuiltinClass {
        match self {
            Self::Compiled(_) => BuiltinClass::Regex,
            Self::Builder { .. } => BuiltinClass::RegexBuilder,
            Self::Match { .. } => BuiltinClass::RegexMatch,
            Self::Captures { .. } => BuiltinClass::RegexCaptures,
        }
    }

    pub(super) fn builtin(&self) -> BuiltinId {
        match self {
            Self::Compiled(_) => BuiltinId::Regex,
            Self::Builder { .. } => BuiltinId::RegexBuilder,
            Self::Match { .. } => BuiltinId::RegexMatch,
            Self::Captures { .. } => BuiltinId::RegexCaptures,
        }
    }

    pub(super) fn external_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                // regexr does not expose compiled engine/cache memory usage.
                Self::Compiled(regex) => {
                    std::mem::size_of::<regexr::Regex>() + regex.as_str().len()
                }
                Self::Builder { pattern_len, .. } => *pattern_len,
                Self::Match { text, .. } => text.len(),
                Self::Captures { text, slots, names } => {
                    text.len()
                        + std::mem::size_of_val(slots.as_ref())
                        + names.capacity() * std::mem::size_of::<(String, Option<Span>)>()
                        + names.keys().map(String::capacity).sum::<usize>()
                }
            }
    }

    fn captures(text: &Arc<str>, regex: &regexr::Regex, captures: regexr::Captures<'_>) -> Self {
        Self::Captures {
            text: text.clone(),
            slots: (0..captures.len())
                .map(|i| captures.get(i).map(|m| (m.start(), m.end())))
                .collect::<Vec<_>>()
                .into(),
            names: Arc::new(
                regex
                    .capture_names()
                    .map(|name| {
                        (
                            name.to_owned(),
                            captures.name(name).map(|m| (m.start(), m.end())),
                        )
                    })
                    .collect(),
            ),
        }
    }
}

impl fmt::Display for RegexObject {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiled(regex) => out.write_str(regex.as_str()),
            Self::Builder { .. } => out.write_str("<RegexBuilder>"),
            Self::Match {
                text,
                span: (start, end),
            } => out.write_str(&text[*start..*end]),
            Self::Captures { slots, .. } => write!(out, "<RegexCaptures length={}>", slots.len()),
        }
    }
}

fn unsigned(value: Value, maximum: u64) -> Result<u64, String> {
    let n = value
        .as_num()
        .ok_or("expected an unsigned integer Number")?;
    if !n.is_finite() || n.fract() != 0.0 || n < 0.0 || n > maximum as f64 {
        return Err(format!("expected an integer between 0 and {maximum}"));
    }
    Ok(n as u64)
}

fn boolean(value: Value) -> Result<bool, String> {
    if !value.is_bool() {
        return Err("expected a Bool".into());
    }
    Ok(value == Value::TRUE)
}

fn string_object(text: &str) -> HeapObject {
    HeapObject::String(NapalmString::new(Arc::<[u8]>::from(text.as_bytes())))
}

fn default_regex_builder(pattern: &str) -> regexr::RegexBuilder {
    // Every Napalm constructor requests native compilation. Explicit builder
    // settings may override this; backend selection remains regexr's decision.
    regexr::RegexBuilder::new(pattern).jit(true)
}

impl Interpreter<'_> {
    fn regex_text(&self, value: Value) -> Result<Arc<str>, String> {
        let Some(HeapObject::String(text)) = self.heap.get(value) else {
            return Err("expected a String".into());
        };
        std::str::from_utf8(text.as_ref())
            .map(Arc::from)
            .map_err(|_| "expected a String containing valid UTF-8".into())
    }

    fn regex_alloc(&mut self, object: RegexObject) -> Value {
        self.heap.alloc(HeapObject::Regex(Box::new(object)))
    }

    fn regex_match(&mut self, text: Arc<str>, span: Option<Span>) -> Value {
        span.map_or(Value::NULL, |span| {
            self.regex_alloc(RegexObject::Match { text, span })
        })
    }

    fn regex_list(&mut self, items: impl IntoIterator<Item = HeapObject>) -> Value {
        let list = self.heap.alloc(HeapObject::List(Vec::new()));
        let root = self.add_root(list);
        for item in items {
            let value = self.heap.alloc(item);
            self.heap.remember_if_reference(list, value);
            let Some(HeapObject::List(values)) = self.heap.get_mut(list) else {
                unreachable!()
            };
            values.push(value);
        }
        self.release_root(root);
        let external = self.heap.get(list).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(list, external);
        self.heap.refresh_jit_view(list);
        list
    }

    fn regex_operation(
        &mut self,
        native: RegexNative,
        receiver: Value,
        args: &[RegisterWord],
    ) -> Result<Value, String> {
        use RegexNative::*;
        let arg = |i: usize| {
            args.get(i)
                .copied()
                .map(Value::from_word)
                .unwrap_or(Value::NULL)
        };
        match native {
            New => {
                let pattern = self.regex_text(arg(0))?;
                let regex = default_regex_builder(&pattern)
                    .build()
                    .map_err(|e| e.to_string())?;
                return Ok(self.regex_alloc(RegexObject::Compiled(Arc::new(regex))));
            }
            BuilderNew => {
                let pattern = self.regex_text(arg(0))?;
                return Ok(self.regex_alloc(RegexObject::Builder {
                    builder: default_regex_builder(&pattern),
                    pattern_len: pattern.len(),
                }));
            }
            Escape => {
                let text = self.regex_text(arg(0))?;
                return Ok(self.heap.alloc(string_object(&regexr::escape(&text))));
            }
            _ => {}
        }
        // Clone only owned native state. No borrowed GC data crosses an allocation.
        let object = match self.heap.get(receiver) {
            Some(HeapObject::Regex(object)) => (**object).clone(),
            _ => return Err("invalid Regex receiver".into()),
        };
        match object {
            RegexObject::Compiled(regex) => {
                match native {
                    AsStr => return Ok(self.heap.alloc(string_object(regex.as_str()))),
                    EngineName => return Ok(self.heap.alloc(string_object(regex.engine_name()))),
                    CaptureNames => {
                        return Ok(self.regex_list(regex.capture_names().map(string_object)));
                    }
                    _ => {}
                }
                let text = self.regex_text(arg(0))?;
                Ok(match native {
                    IsMatch => Value::from_bool(regex.is_match(&text)),
                    TryIsMatch => {
                        Value::from_bool(regex.try_is_match(&text).map_err(|e| e.to_string())?)
                    }
                    Find | TryFind => {
                        let found = if native == TryFind {
                            regex.try_find(&text).map_err(|e| e.to_string())?
                        } else {
                            regex.find(&text)
                        };
                        let span = found.map(|m| (m.start(), m.end()));
                        self.regex_match(text, span)
                    }
                    FindIter => self.regex_list(regex.find_iter(&text).map(|m| {
                        HeapObject::Regex(Box::new(RegexObject::Match {
                            text: text.clone(),
                            span: (m.start(), m.end()),
                        }))
                    })),
                    Captures | TryCaptures => {
                        let found = if native == TryCaptures {
                            regex.try_captures(&text).map_err(|e| e.to_string())?
                        } else {
                            regex.captures(&text)
                        };
                        found.map_or(Value::NULL, |c| {
                            self.regex_alloc(RegexObject::captures(&text, &regex, c))
                        })
                    }
                    CapturesIter => self.regex_list(regex.captures_iter(&text).map(|c| {
                        HeapObject::Regex(Box::new(RegexObject::captures(&text, &regex, c)))
                    })),
                    Replace | ReplaceAll => {
                        let replacement = self.regex_text(arg(1))?;
                        let result = if native == Replace {
                            regex.replace(&text, &replacement)
                        } else {
                            regex.replace_all(&text, &replacement)
                        };
                        self.heap.alloc(string_object(&result))
                    }
                    _ => return Err("invalid Regex operation".into()),
                })
            }
            RegexObject::Builder {
                builder,
                pattern_len,
            } => {
                if native == Build {
                    let regex = builder.build().map_err(|e| e.to_string())?;
                    return Ok(self.regex_alloc(RegexObject::Compiled(Arc::new(regex))));
                }
                let builder = match native {
                    Jit => builder.jit(boolean(arg(0))?),
                    OptimizePrefixes => builder.optimize_prefixes(boolean(arg(0))?),
                    BacktrackLimit => {
                        builder.backtrack_limit(unsigned(arg(0), 9_007_199_254_740_991)?)
                    }
                    SizeLimit => builder.size_limit(unsigned(arg(0), u32::MAX as u64)? as u32),
                    NestLimit => builder.nest_limit(unsigned(arg(0), u32::MAX as u64)? as u32),
                    _ => return Err("invalid RegexBuilder operation".into()),
                };
                let Some(HeapObject::Regex(object)) = self.heap.get_mut(receiver) else {
                    unreachable!()
                };
                **object = RegexObject::Builder {
                    builder,
                    pattern_len,
                };
                Ok(receiver)
            }
            RegexObject::Match {
                text,
                span: (start, end),
            } => Ok(match native {
                Start => Value::from_num(start as f64),
                End => Value::from_num(end as f64),
                Len => Value::from_num((end - start) as f64),
                IsEmpty => Value::from_bool(start == end),
                AsStr => self.heap.alloc(string_object(&text[start..end])),
                AsBytes => self.heap.alloc(HeapObject::Bytes(NapalmBytes::from_slice(
                    &text.as_bytes()[start..end],
                ))),
                Range => self.heap.alloc(HeapObject::Range {
                    from: start as f64,
                    to: end as f64,
                    inclusive: false,
                }),
                _ => return Err("invalid RegexMatch operation".into()),
            }),
            RegexObject::Captures { text, slots, names } => {
                match native {
                    Len => return Ok(Value::from_num(slots.len() as f64)),
                    IsEmpty => return Ok(Value::from_bool(slots.is_empty())),
                    _ => {}
                }
                let indexed_span = |value| -> Result<Option<Span>, String> {
                    let index = unsigned(value, 9_007_199_254_740_991)?;
                    // An out-of-range index stays missing on 32-bit hosts too.
                    Ok(usize::try_from(index)
                        .ok()
                        .and_then(|index| slots.get(index))
                        .copied()
                        .flatten())
                };
                let span = match native {
                    Get => indexed_span(arg(0))?,
                    Name => names
                        .get(self.regex_text(arg(0))?.as_ref())
                        .copied()
                        .flatten(),
                    Index => {
                        if arg(0).is_num() {
                            indexed_span(arg(0))?
                        } else {
                            names
                                .get(self.regex_text(arg(0))?.as_ref())
                                .copied()
                                .flatten()
                        }
                    }
                    _ => return Err("invalid RegexCaptures operation".into()),
                };
                if native == Index {
                    let (start, end) =
                        span.ok_or("capture group is missing or did not participate")?;
                    Ok(self.heap.alloc(string_object(&text[start..end])))
                } else {
                    Ok(self.regex_match(text, span))
                }
            }
        }
    }

    pub(super) fn execute_regex_native(
        &mut self,
        op: &PreparedOp,
        native: RegexNative,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) {
        self.mark_gc_point(op);
        self.maybe_collect();
        match self.regex_operation(native, Value::from_word(receiver), args) {
            Ok(value) => unsafe {
                self.set_reg(op.a, value.word());
                self.advance(op);
            },
            Err(error) => self.fail_at(op, format!("Regex: {error}")),
        }
    }
}
