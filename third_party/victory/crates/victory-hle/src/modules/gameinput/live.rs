//! `VICTORY_INPUT_LIVE=<file>`: pad presses appended to a file while the
//! title runs, for an outside driver (a start-screen pilot) that must not
//! take the focus or synthesize host input. Read on the poll thread every
//! 100 ms; what it holds goes into every pad the title sees (host slots and
//! the keyboard's stand-in) whether or not the window has focus, and A, B,
//! Menu, View and the d-pad also hold their stand-in keys for titles that
//! listen to the keyboard.
//!
//! One command per line, ended by `\n` (`\r\n` too), case ignored:
//!
//! ```text
//! line     := empty | "#" text | press | release
//! press    := name [":" duration]            # no duration: 0.3 s
//! release  := "release" [":" name]           # everything, or that name
//! duration := seconds ["s"]                  # 0 < s <= 600
//!           | count "f"                      # 1..100000 frames presented from the press
//!           | "hold"                         # until a release, or the caps
//! name     := a b x y start|menu back|view lb rb ls rs up down left right lt rt rsup rsdown
//! ```
//!
//! An open hold ends after `VICTORY_INPUT_LIVE_MAX_S` seconds (30) or
//! `VICTORY_INPUT_LIVE_MAX_FRAMES` presented frames (1800), whichever comes
//! first; a frame count ends after 600 s if nothing is presented. What the
//! file holds when input starts is skipped (no replay of an earlier
//! session); a file made later counts from its first byte. A file that
//! shrinks is read again from the start. NUL and non-ASCII bytes are
//! dropped, so UTF-8 (BOM or not) and UTF-16 LE files both work. Each end
//! is logged with the frames presented meanwhile:
//! `release b@594.2 after 3.2 s, 7 frames presented (live, hold, release line)`.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, RIGHT, RIGHT_SHOULDER, RIGHT_STICK, UP,
    VIEW, X, Y,
};

const READ_EVERY: Duration = Duration::from_millis(100);
const DEFAULT_S: f64 = 0.3;
const MAX_SECONDS: f64 = 600.0;
const MAX_COUNT: u64 = 100_000;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Target {
    Button(u32),
    LeftTrigger,
    RightTrigger,
    /// Right stick up (+1) or down (-1).
    RightStick(i8),
}

const NAMES: [(&str, Target); 20] = [
    ("a", Target::Button(A)),
    ("b", Target::Button(B)),
    ("x", Target::Button(X)),
    ("y", Target::Button(Y)),
    ("start", Target::Button(MENU)),
    ("menu", Target::Button(MENU)),
    ("back", Target::Button(VIEW)),
    ("view", Target::Button(VIEW)),
    ("lb", Target::Button(LEFT_SHOULDER)),
    ("rb", Target::Button(RIGHT_SHOULDER)),
    ("ls", Target::Button(LEFT_STICK)),
    ("rs", Target::Button(RIGHT_STICK)),
    ("up", Target::Button(UP)),
    ("down", Target::Button(DOWN)),
    ("left", Target::Button(LEFT)),
    ("right", Target::Button(RIGHT)),
    ("lt", Target::LeftTrigger),
    ("rt", Target::RightTrigger),
    ("rsup", Target::RightStick(1)),
    ("rsdown", Target::RightStick(-1)),
];

/// The stand-in key each button also holds (virtual key), for keyboard listeners.
const KEYS: [(u32, u8); 8] = [
    (A, 0x0D),
    (B, 0x08),
    (MENU, 0x1B),
    (VIEW, 0x09),
    (UP, 0x26),
    (DOWN, 0x28),
    (LEFT, 0x25),
    (RIGHT, 0x27),
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Length {
    Seconds(f64),
    Frames(u64),
    Hold,
}

#[derive(Debug, PartialEq)]
enum Command {
    Nothing,
    Press(&'static str, Target, Length),
    Release(Option<&'static str>),
}

fn name(text: &str) -> Option<(&'static str, Target)> {
    NAMES.iter().copied().find(|(n, _)| *n == text)
}

fn parse(line: &str) -> Result<Command, ()> {
    let line = line.trim().to_ascii_lowercase();
    if line.is_empty() || line.starts_with('#') {
        return Ok(Command::Nothing);
    }
    let (head, tail) = match line.split_once(':') {
        Some((h, t)) => (h.trim(), Some(t.trim())),
        None => (line.as_str(), None),
    };
    if head == "release" {
        return match tail {
            None => Ok(Command::Release(None)),
            Some(t) => name(t).map(|(n, _)| Command::Release(Some(n))).ok_or(()),
        };
    }
    let (n, target) = name(head).ok_or(())?;
    let length = match tail {
        None => Length::Seconds(DEFAULT_S),
        Some("hold") => Length::Hold,
        Some(t) if t.ends_with('f') => {
            let count: u64 = t[..t.len() - 1].trim().parse().map_err(|_| ())?;
            if !(1..=MAX_COUNT).contains(&count) {
                return Err(());
            }
            Length::Frames(count)
        }
        Some(t) => {
            let s: f64 = t
                .strip_suffix('s')
                .unwrap_or(t)
                .trim()
                .parse()
                .map_err(|_| ())?;
            if !(s > 0.0 && s <= MAX_SECONDS) {
                return Err(());
            }
            Length::Seconds(s)
        }
    };
    Ok(Command::Press(n, target, length))
}

struct Press {
    name: &'static str,
    target: Target,
    length: Length,
    began: Instant,
    /// Seconds since input started, for the log.
    at: f64,
    frame: u64,
}

/// What the live presses hold now.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Out {
    pub buttons: u32,
    pub left_trigger: f32,
    pub right_trigger: f32,
    /// Right stick vertical, -1 to 1 (0: untouched).
    pub right_y: f32,
    /// Stand-in keys (virtual keys) for the buttons above.
    pub keys: Vec<u8>,
}

pub(super) struct Live {
    path: PathBuf,
    offset: u64,
    partial: Vec<u8>,
    last_read: Option<Instant>,
    presses: Vec<Press>,
    max_s: f64,
    max_frames: u64,
}

impl Live {
    /// From `VICTORY_INPUT_LIVE` (and its caps); `None` when unset.
    pub(super) fn from_env() -> Option<Live> {
        let path = std::env::var_os("VICTORY_INPUT_LIVE").filter(|p| !p.is_empty())?;
        let max_s = std::env::var("VICTORY_INPUT_LIVE_MAX_S")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|s: &f64| *s > 0.0)
            .unwrap_or(30.0);
        let max_frames = std::env::var("VICTORY_INPUT_LIVE_MAX_FRAMES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n: &u64| *n > 0)
            .unwrap_or(1800);
        let live = Live::new(PathBuf::from(path), max_s, max_frames);
        tracing::info!(
            "gameinput: VICTORY_INPUT_LIVE: reading lines appended to {} every 100 ms",
            live.path.display()
        );
        tracing::info!(
            "gameinput: an open hold (b:hold) ends after {max_s} s or {max_frames} presented frames"
        );
        Some(live)
    }

    pub(super) fn new(path: PathBuf, max_s: f64, max_frames: u64) -> Live {
        // What is there already is skipped.
        let offset = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        Live {
            path,
            offset,
            partial: Vec::new(),
            last_read: None,
            presses: Vec::new(),
            max_s,
            max_frames,
        }
    }

    /// New complete lines of the file, at most every 100 ms.
    fn lines(&mut self, now: Instant) -> Vec<String> {
        if self
            .last_read
            .is_some_and(|t| now.saturating_duration_since(t) < READ_EVERY)
        {
            return Vec::new();
        }
        self.last_read = Some(now);
        let Ok(mut file) = std::fs::File::open(&self.path) else {
            return Vec::new();
        };
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        if len < self.offset {
            tracing::info!(
                "gameinput: live: {} shrank, reading it from the start",
                self.path.display()
            );
            self.offset = 0;
            self.partial.clear();
        }
        if len == self.offset {
            return Vec::new();
        }
        let mut bytes = Vec::new();
        if file.seek(SeekFrom::Start(self.offset)).is_err() || file.read_to_end(&mut bytes).is_err()
        {
            return Vec::new();
        }
        self.offset += bytes.len() as u64;
        self.partial
            .extend(bytes.into_iter().filter(|&b| b != 0 && b < 0x80));
        let mut out = Vec::new();
        while let Some(i) = self.partial.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.partial.drain(..=i).collect();
            out.push(
                String::from_utf8_lossy(&line)
                    .trim_end_matches(['\n', '\r'])
                    .to_string(),
            );
        }
        out
    }

    fn end(&self, p: &Press, now: Instant, frames: u64, why: &str) {
        let held = now.saturating_duration_since(p.began).as_secs_f64();
        let length = match p.length {
            Length::Seconds(s) => format!("{s:.1} s"),
            Length::Frames(n) => format!("{n}f"),
            Length::Hold => "hold".to_string(),
        };
        let why = if why.is_empty() {
            String::new()
        } else {
            format!(", {why}")
        };
        tracing::info!(
            "gameinput: release {}@{:.1} after {held:.1} s, {} frames presented (live, {length}{why})",
            p.name,
            p.at,
            frames.saturating_sub(p.frame),
        );
    }

    /// Reads new lines, ends what is over, and says what is held now.
    /// `t`: seconds since input started; `frames`: frames presented so far.
    pub(super) fn poll(&mut self, now: Instant, t: f64, frames: u64) -> Out {
        for line in self.lines(now) {
            match parse(&line) {
                Ok(Command::Nothing) => {}
                Ok(Command::Press(name, target, length)) => {
                    let shown = match length {
                        Length::Seconds(s) => format!("{s:.1}"),
                        Length::Frames(n) => format!("{n}f"),
                        Length::Hold => "hold".into(),
                    };
                    tracing::info!("gameinput: live press {name}@{t:.1}:{shown}");
                    self.presses.push(Press {
                        name,
                        target,
                        length,
                        began: now,
                        at: t,
                        frame: frames,
                    });
                }
                Ok(Command::Release(which)) => {
                    tracing::info!(
                        "gameinput: live release {} at {t:.1} s",
                        which.unwrap_or("all")
                    );
                    let (gone, kept): (Vec<_>, Vec<_>) = self
                        .presses
                        .drain(..)
                        .partition(|p| which.is_none_or(|n| n == p.name));
                    for p in &gone {
                        self.end(p, now, frames, "release line");
                    }
                    self.presses = kept;
                }
                Err(()) => tracing::warn!(
                    "gameinput: live: unknown line '{}' (button[:seconds|:<n>f|:hold], release[:button] or # comment)",
                    line.chars().take(60).collect::<String>()
                ),
            }
        }
        let (max_s, max_frames) = (self.max_s, self.max_frames);
        let over = |p: &Press| -> Option<&'static str> {
            let held = now.saturating_duration_since(p.began).as_secs_f64();
            let shown = frames.saturating_sub(p.frame);
            match p.length {
                Length::Seconds(s) => (held >= s).then_some(""),
                Length::Frames(n) => {
                    if shown >= n {
                        Some("")
                    } else {
                        (held >= MAX_SECONDS).then_some("cap")
                    }
                }
                Length::Hold => (held >= max_s || shown >= max_frames).then_some("cap"),
            }
        };
        let mut kept = Vec::with_capacity(self.presses.len());
        for p in std::mem::take(&mut self.presses) {
            match over(&p) {
                Some(why) => self.end(&p, now, frames, why),
                None => kept.push(p),
            }
        }
        self.presses = kept;
        let mut out = Out::default();
        for p in &self.presses {
            match p.target {
                Target::Button(bit) => out.buttons |= bit,
                Target::LeftTrigger => out.left_trigger = 1.0,
                Target::RightTrigger => out.right_trigger = 1.0,
                Target::RightStick(d) => out.right_y = f32::from(d),
            }
        }
        out.keys = KEYS
            .iter()
            .filter(|(bit, _)| out.buttons & bit != 0)
            .map(|&(_, vk)| vk)
            .collect();
        out
    }
}

/// Adds what the live presses hold to a pad reading.
pub(super) fn apply(pad: &mut super::PadState, live: &Out) {
    pad.buttons |= live.buttons;
    pad.left_trigger = pad.left_trigger.max(live.left_trigger);
    pad.right_trigger = pad.right_trigger.max(live.right_trigger);
    if live.right_y != 0.0 {
        pad.right_y = live.right_y;
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn append(path: &std::path::Path, text: &[u8]) {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        f.write_all(text).unwrap();
    }

    #[test]
    fn grammar() {
        assert_eq!(parse(""), Ok(Command::Nothing));
        assert_eq!(parse("  # note"), Ok(Command::Nothing));
        assert_eq!(
            parse("A"),
            Ok(Command::Press("a", Target::Button(A), Length::Seconds(0.3)))
        );
        assert_eq!(
            parse(" b : 3s "),
            Ok(Command::Press("b", Target::Button(B), Length::Seconds(3.0)))
        );
        assert_eq!(
            parse("down:20f"),
            Ok(Command::Press(
                "down",
                Target::Button(DOWN),
                Length::Frames(20)
            ))
        );
        assert_eq!(
            parse("y:hold"),
            Ok(Command::Press("y", Target::Button(Y), Length::Hold))
        );
        assert_eq!(
            parse("start"),
            Ok(Command::Press(
                "start",
                Target::Button(MENU),
                Length::Seconds(0.3)
            ))
        );
        assert_eq!(
            parse("rt:0.4"),
            Ok(Command::Press(
                "rt",
                Target::RightTrigger,
                Length::Seconds(0.4)
            ))
        );
        assert_eq!(parse("release"), Ok(Command::Release(None)));
        assert_eq!(parse("release:Y"), Ok(Command::Release(Some("y"))));
        for bad in [
            "zz",
            "a:0",
            "a:601",
            "a:0f",
            "a:100001f",
            "a:x",
            "release:zz",
            "?",
        ] {
            assert_eq!(parse(bad), Err(()), "{bad}");
        }
    }

    #[test]
    fn presses_hold_end_and_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.txt");
        append(&path, b"a:hold\n"); // there before input starts: skipped
        let mut live = Live::new(path.clone(), 1.0, 50);
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        assert_eq!(live.poll(at(0), 0.0, 0).buttons, 0);

        // seconds, CRLF, a comment, an unknown line, a line written in two parts
        append(&path, b"b:1.2\r\n# x\r\nzz\nx:");
        assert_eq!(live.poll(at(200), 0.2, 2).buttons, B);
        assert_eq!(live.poll(at(1000), 1.0, 10).keys, vec![0x08]);
        append(&path, b"0.5\n");
        assert_eq!(live.poll(at(1300), 1.3, 13).buttons, B | X);
        assert_eq!(live.poll(at(1500), 1.5, 15).buttons, X);
        assert_eq!(live.poll(at(1900), 1.9, 19).buttons, 0);

        // frames: held until 20 frames are presented, however long that takes
        append(&path, b"down:20f\n");
        assert_eq!(live.poll(at(2000), 2.0, 100).buttons, DOWN);
        assert_eq!(live.poll(at(2900), 2.9, 119).buttons, DOWN);
        assert_eq!(live.poll(at(2950), 2.95, 120).buttons, 0);

        // hold, released by name; the frame cap (50) and the time cap (1 s)
        append(&path, b"y:hold\nrt:hold\n");
        let out = live.poll(at(3000), 3.0, 200);
        assert_eq!((out.buttons, out.right_trigger), (Y, 1.0));
        append(&path, b"release:Y\n");
        let out = live.poll(at(3200), 3.2, 210);
        assert_eq!((out.buttons, out.right_trigger), (0, 1.0));
        assert_eq!(live.poll(at(3300), 3.3, 249).right_trigger, 1.0);
        assert_eq!(live.poll(at(3400), 3.4, 250).right_trigger, 0.0);
        append(&path, b"lb:hold\n");
        assert_eq!(live.poll(at(4000), 4.0, 300).buttons, LEFT_SHOULDER);
        assert_eq!(live.poll(at(5000), 5.0, 301).buttons, 0);

        // release everything
        append(&path, b"a:hold\nmenu:hold\n");
        assert_eq!(live.poll(at(5200), 5.2, 310).keys, vec![0x0D, 0x1B]);
        append(&path, b"release\n");
        assert_eq!(live.poll(at(5400), 5.4, 312), Out::default());

        // read at most every 100 ms
        append(&path, b"a:5\n");
        assert_eq!(live.poll(at(5450), 5.45, 313).buttons, 0);
        assert_eq!(live.poll(at(5500), 5.5, 313).buttons, A);
        append(&path, b"release\n");
        live.poll(at(5600), 5.6, 314);

        // a file rewritten shorter, in UTF-16 LE with a BOM, is read from the start
        let text: Vec<u8> = "\u{feff}b:hold\r\n"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        std::fs::write(&path, &text).unwrap();
        assert_eq!(live.poll(at(5700), 5.7, 315).buttons, B);
    }

    #[test]
    fn a_file_made_later_counts_from_its_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("later.txt");
        let mut live = Live::new(path.clone(), 30.0, 1800);
        let t0 = Instant::now();
        assert_eq!(live.poll(t0, 0.0, 0).buttons, 0);
        append(&path, b"a\n");
        assert_eq!(
            live.poll(t0 + Duration::from_millis(100), 0.1, 1).buttons,
            A
        );
        assert_eq!(
            live.poll(t0 + Duration::from_millis(400), 0.4, 4).buttons,
            0
        );
    }

    #[test]
    fn apply_adds_to_a_pad() {
        let mut pad = super::super::PadState {
            buttons: X,
            left_trigger: 0.2,
            ..Default::default()
        };
        let out = Out {
            buttons: A,
            left_trigger: 1.0,
            right_y: -1.0,
            ..Default::default()
        };
        apply(&mut pad, &out);
        assert_eq!(
            (pad.buttons, pad.left_trigger, pad.right_y),
            (X | A, 1.0, -1.0)
        );
    }
}
