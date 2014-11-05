#![feature(overloaded_calls, unboxed_closures)]

//! Library for easily generating task indices and running tasks for N-dimensional workloads
//! (N in {1,2,3}).
//! See `run` for the simplest way to use the library.
//!
//! # Note about ICE
//!
//! Due to `rustc` issue #17021, compiling this library crashes the compiler if debug info is
//! enabled. See the
//! [Cargo manifest documentation](http://crates.io/manifest.html#the-[profile.*]-sections)
//! for information on how to disable debug info.

use std::sync::Arc;
use std::sync::atomic::AtomicUint;
use std::sync::atomic::SeqCst;

/// A type describing an N-dimensional grid of tasks.
///
/// Tasksets are intended to be put into an `Arc`
/// and then cloned to each thread processing tasks.
pub struct TaskSet<T> {
    next: AtomicUint,
    size: T
}

/// Trait for types that can act as task grid coordinates or size.
pub trait Dim: Copy+Send+Sync+'static {
    /// Called to generate a task grid coordinate from a number. `size` is the grid size passed to
    /// `run` or `TaskSet::new` and `id` is an increasing integer. Return `None` if there are no
    /// more tasks to run.
    fn from_task_id(id: uint, size: Self) -> Option<Self>;
}

impl Dim for uint {
    fn from_task_id(id: uint, size: uint) -> Option<uint> {
        if id < size {
            Some(id)
        } else {
            None
        }
    }
}

impl Dim for (uint, uint) {
    fn from_task_id(id: uint, (w, h): (uint, uint)) -> Option<(uint, uint)> {
        if id < w*h {
            Some((id % w, id / w))
        } else {
            None
        }
    }
}

impl Dim for (uint, uint, uint) {
    fn from_task_id(id: uint, (w, h, d): (uint, uint, uint)) -> Option<(uint, uint, uint)> {
        if id < w*h*d {
            Some((id % w,id / w % h,id / (w*h)))
        } else {
            None
        }
    }
}

impl<T: Dim> TaskSet<T> {
    /// Create a new TaskSet with `n_tasks` tasks. `n_tasks` can be either an `uint` or a
    /// 2-tuple or 3-tuple of `uint`s, producing a 1d, 2d or 3d grid of tasks.
    pub fn new(n_tasks: T) -> TaskSet<T> {
        TaskSet {
            next: AtomicUint::new(0),
            size: n_tasks
        }
    }

    /// Get ID of next task. The task ID type will correspond to the type passed to `new`.
    /// For example, if the taskset size is `(2u, 2u)` then the first returned task ID will be
    /// `(0u, 0u)` and last `(1u, 1u)`.
    pub fn get(&self) -> Option<T> {
        let id = self.next.fetch_add(1, SeqCst);
        Dim::from_task_id(id, self.size)
    }
}

/// Runs tasks in grid specified by `size` on `num_cpus()` threads.
///
/// Tasks are run in parallel on as many threads as the
/// machine has CPUs (See `std::os::num_cpus()`). Execution is blocked until all tasks have
/// completed.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUint, SeqCst};
///
/// let c = Arc::new(AtomicUint::new(0));
/// run((2u, 2u), c.clone(), |&: (x,y): (uint, uint), c: Arc<AtomicUint>| {
///   d.fetch_add(2*x + y, SeqCst);
/// });
/// assert_eq!(c.load(SeqCst), 6);
/// ```
pub fn run<T: Dim, F: Fn<(T,D),()>+Send+Sync, D: Clone+Send>(size: T, data: D, f: F) {
    run_on_n(size, std::os::num_cpus(), data, f)
}

/// Runs tasks in grid specified by `size` on specified number of threads.
///
/// Tasks are run in parallel on `threads` threads.
/// Execution is blocked until all tasks have completed.
pub fn run_on_n<T: Dim, F: Fn<(T,D),()>+Send+Sync, D: Clone+Send>(size: T, threads: uint, data: D, f: F) {
    let ts = Arc::new(TaskSet::new(size));
    let f = Arc::new(f);
    let (completion_tx, completion_rx) = std::comm::channel();
    for _ in range(0, threads) {
        let ts = ts.clone();
        let f = f.clone();
        let d = data.clone();
        let completion_tx = completion_tx.clone();
        std::task::spawn(proc() {
            loop {
                match ts.get() {
                    Some(id) => (*f)(id,d.clone()),
                    None     => break
                }
            }
            completion_tx.send(());
        });
    }
    for _ in range(0, threads) {
        completion_rx.recv();
    }
}

#[test]
fn test_1d() {
    let ts = TaskSet::new(2u);
    assert_eq!(ts.get(), Some(0));
    assert_eq!(ts.get(), Some(1));
    assert_eq!(ts.get(), None);
}

#[test]
fn test_2d() {
    let ts = TaskSet::new((2u, 2u));
    assert_eq!(ts.get(), Some((0,0)));
    assert_eq!(ts.get(), Some((1,0)));
    assert_eq!(ts.get(), Some((0,1)));
    assert_eq!(ts.get(), Some((1,1)));
    assert_eq!(ts.get(), None);
}

#[test]
fn test_3d() {
    let ts = TaskSet::new((2u, 2u, 2u));
    assert_eq!(ts.get(), Some((0,0,0)));
    assert_eq!(ts.get(), Some((1,0,0)));
    assert_eq!(ts.get(), Some((0,1,0)));
    assert_eq!(ts.get(), Some((1,1,0)));
    assert_eq!(ts.get(), Some((0,0,1)));
    assert_eq!(ts.get(), Some((1,0,1)));
    assert_eq!(ts.get(), Some((0,1,1)));
    assert_eq!(ts.get(), Some((1,1,1)));
    assert_eq!(ts.get(), None);
}

#[test]
fn test_run_n() {
    let c = Arc::new(AtomicUint::new(0));

    run_on_n(3u, 2, c.clone(), |&: id, c: Arc<AtomicUint>| {
        c.fetch_add(id, SeqCst);
    });

    assert_eq!(c.load(SeqCst), 3);
}

#[test]
fn test_doc1() {
    let c = Arc::new(AtomicUint::new(0));
    run((2u, 2u), c.clone(), |&: (x,y): (uint, uint), c: Arc<AtomicUint>| {
        c.fetch_add(2*x + y, SeqCst);
    });
    assert_eq!(c.load(SeqCst), 6);
}
