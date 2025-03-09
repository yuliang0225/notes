# Study Rust
#Study
## Ultimate Rust Crash Course
- https://www.udemy.com/course/ultimate-rust-crash-course/
- https://github.com/CleanCut/ultimate_rust_crash_course/#exercises
- [Using config files - Command Line Applications in Rust](https://rust-cli.github.io/book/in-depth/config-files.html)
- https://github.com/kyclark/command-line-rust/tree/clap_v2?tab=readme-ov-file
- [The Rust Programming Language 日本語版 - The Rust Programming Language 日本語版](https://doc.rust-jp.rs/book-ja/title-page.html)
## Section 2: Fundamentals
### Cargo
- package manager
  - search install manage packages
  - test
  - docs generator
  - No more make files!
- File
  - Cargo.toml
    - semantic versioning
    - Target debug and run
  - ![](Study%20Rust/%E6%88%AA%E5%B1%8F2024-07-31%2023.40.51.png)<!-- {"width":492} -->
  - src/main.rs
  - `cargo run`
    - --release —> more fast
### Variables
- `let a = 32;`
  - Safety, Concurrency, Speed, immutable
- `let mut bunnies = 32;`
  - mutable
- `const A_B: f64 = 3.2;`
  - Const can be outside function
  - Really fast
### Scope
- Variables have a scope, which is the place in the code that you are allowed to use them.
- Scope —> Between { and }.
### Memory safety
- No init, can not be used.
### Function
- Functions do not have to appear in the file before code that calls
- Tail expression
- macro
### Module system
- `hello::greet()`
  - `pub fn greet()`
- `use hello::greet`
- `use std::collections::HashMap`
  - rust std
  - crates.io 
### Scalar types
- Integer: unsigned and signed u8 vs. i8, usize and isize
  - Decimal, Hex, Octal, Binary, Byte (u8 only)
- Float: f32 and f64
  - IEEE-745
- `let x = 5u16` or `let x = 5_u16`
- Boolean
- char 4 bytes usc-4
### Compound types
- Tuple
  - `let info = (1, 3.3, 999) `
  - `info.0`
  - Max 12
- Arrary
  - Same type `let buf: [u8; 3] = [1, 2, 3]`, `buf[0]`
  - We also use vectors to instead of arrays
### Control flow
- `if num == 5 {} else if num == 4 {} else {}`
```rust
mut msg = if num == 5 {"five"} else if nums == 4 {"four"} else {"other"};
// same type!
```
- `loop { break; }`
  - continue; add label `'bob`
  - `while dizzy() {//do stuff}`
```rust
for num in [7, 8, 9].iter() {
	// do stuff
}

for (x,y) in array.iter() {
	// do stuff
}

for num in 0..50 {
	// do stuff
}
```
### Strings
- 6 types of strings in rust
- str and &str
  - Can not be modified. It is a borrow.
  - Pointer of start and Len —> &str
- String
  - Can be modified.
  - `let masg = "ab".to_string();`
  - `let msg = String::from("ab");`
  - Cannot be indexed by character position.
    - Not only English
    - unicode scalars, graphemes
    - `word.bytes();` —> UTF8 byte vector
    - `graphemes(my_string, true)`
    - `.nth(3)`
- Valid UTF-8
### Ownership
- Systems Programing
- Make those crazy safety guarantees possible and makes Rust so different from other
- Ownership is what makes all those informative compiler error messages possible and necessary.
- 3 rules:
  1. Each value has an owner
  2. Only one owner
  3. Value gets dropped if its owner goes out of scope.
```rust
let s1 = String::from("abc"):
let s2 = s1;
println!("{}", s1) // Error! Owner has been moved to s2.

let s1 = String::from("abc"):
let s2 = s1.clone();
println!("{}", s1) // OK Stak and heap --> Deep copy

let mut s1 = String::from("abc");
s1 = do_stuff(s1)

fn do_stuff (s: String) -> String {
	// do stuff
	s
}
```
### References and Borrowing 
- Reference to a type
  - Borrow the value from s1
  - The reference, not the value, gets moved into the function.
  - Lifetimes —> Reference must always be valid
    - You can never point to null
  - Mutable reference and mutable value can be modified.
```rust
let mut s1 = String::from("abc");
do_stuff(&mut s1);

fn do_stuff(s: &mut String) {
	// do stuff
	s.insert_str(0, "Hi, ");
	// We can use *s = String::from("Replancement"); to replace the entire value.
}
```
- **You can have either exactly one mutable reference or any number of immutable references.**
- Compilers —> informative error
### Structs
- Data files, methods and associated functions
```rust
struct RedFox {
	enemy: bool,
	life: u8,
};

impl RedFox {
	fn new() -> Self {
		Self {
			enemy: true,
			life: 70,
		}
	}
	fn move(self)
	fn boeeow(&self)
	fn mut_borrow(&mut self)
}

let fox = RedFox::new();
```
- Class inheritance
- No structure inheritance
  - Because they chose a better way to solve the problem wish inheritance solved: Traits.
### Traits
- [Traits: Defining Shared Behavior - The Rust Programming Language](https://doc.rust-lang.org/book/ch10-02-traits.html)
- Similar to interface in other languages
```rust
trait Noisy {
	fn get_noise(&self) -> &self;
}

impl Noisy for RedFox {
	fn get_noise(&self) -> &str {"Mewo?"}
}
```
- You can implement your traits on any types from anywhere including built-ins or types.
- Special traits: Copy
  - Simple primitive types, integers floats and booleans implement Copy.
  - Heap —> No copy
- No fileds in traits.
### Collections
- [Common Collections - The Rust Programming Language](https://doc.rust-lang.org/book/ch08-00-common-collections.html)
- STD library
- `Vec<T>`
  - A generic collection that holds a bunch of one type
- `HashMap<K, V>`
  - inert, look up and remove values by key in constant time
- VecDeque: Add and remove more fast from the front and the back
- LinkedList, HashSet, BInaryHeap
- BTreeMap, BTreeSet —> Always be sorted.
```rust
let mut v: Vec<i32> = Vector::new();

v.push(2);
let x = v.pop();
println!("{}", v[1);

let mut v = vector![2, 4, 6];

// HashMap
let mut h: HashMap<u8, bool> = HashMap::new();
h.insert(5, true);
let have_five = h.remove(&5).unwrap();
```
### Enums
- Union in C but much better
- Generic type (T)
```rust
enum Color {
	Red,
	Green,
	Bluee
}

enum DispenserItem {
	Empty,
	Ammo(u8),
	Things(String, i32),
	Place {x: i32, y: i32},
}

use DispenserItem::*;
let item = Place{32, 46};

impl DispenserItem {
	fn display(&self) {}
}

enum Option<T> {
	Some(T),
	None,
}


match my_var {
	Some(x) => {
		println!("value is {}", x);
	},
	None => {
		println!("no value");
	},
	_ => {
		println!("who cares");
	},
}
```
### Option and Result
- Option: whenever something might be absent.
```rust
let mut x = None;
x = Some(5);
x.is_some(); // true
x.is_none(); //false
```
- Result: Whenever something might have a useful result or might have an error.
  - IO module
  - Rust strongly encourage you to look at all possible errors and make a conscious choice what to do with each one.
```rust
#[must_use]
enum Result<T, E> {
	Ok(T),
	Err(E),
};

use std::fs::File;

fn main() {
	let res = File::open("foo");
	match res {
		Ok(f) => {},
		Err(e) => {},
	}
}
```
### Closures
- A closure is an anonymous function that can borrow or capture some data from the scope it is nested in.
- `|x, y| {x+y}`
- The type of the arguments and the return value
  - are all inferred from how you use the arguments
  - and what you return
- A closure will borrow a reference to values in the enclosing scope.
  - Functional-style programming, use closure.
```rust
let mut b = vec![2, 4, 6];

v.iter().map(|x| x*3).filter(|x| *x > 10).fold(0, |acc, x| acc+x);
```
### Threads
- Threads are fantastic tool when you need to use CPU and memory concurrently, because they can run simultaneously on multi cores, and actually accomplish more work!
```rust
use std::thread;

fn main() {
	let handle = thread::spawn(move || {
		// do stuff in a child thread
	});

	// do stuff simultaneously in the main thread
	// wait untill thread has exited
	handle.join().unwrap();
}
```
### Tree value compre
- [2236. 判断根结点是否等于子结点之和](https://leetcode.cn/problems/root-equals-sum-of-children/?envType=study-plan-v2&envId=primers-list)
```rust
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
// 
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
use std::rc::Rc;
use std::cell::RefCell;
impl Solution {
    pub fn check_tree(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let node = root.as_ref().unwrap().borrow();
        let left = node.left.as_ref().unwrap().borrow();
        let right = node.right.as_ref().unwrap().borrow();

        return node.val == left.val + right.val
    }
}
```


---
## Atcoder
- [Rustで競技プログラミングの入力をスッキリ記述するマクロ](https://qiita.com/tanakh/items/0ba42c7ca36cd29d0ac8)
```rust
input!{
    n: usize,
    v: [i32; n],
}

input!{
    n: usize,
    m: usize,
    mat: [[i32; m]; n],
}

input!{
    cs: chars, // トークンを Vec<char> として読む
}

input!{
    h: usize,
    w: usize,
    board: [chars; h],
}

input!{
    n: usize, // ノード数
    m: usize, // 枝数
    edges: [(usize1, usize1); m], // usize1 は 1-origin to 0-origin変換を行う
}
```
