# Cpp study
#Study
[Hands-On Machine Learning with C++.pdf](Cpp%20study/Hands-On%20Machine%20Learning%20with%20C++.pdf)<!-- {"embed":"true"} -->
## C++ basic
* [cppreference.com](https://en.cppreference.com/w/)
* [Harvard CS197: AI Research Experiences – The Course Book](https://docs.google.com/document/d/1uvAbEhbgS_M-uDMTzmOWRlYxqCkogKRXdbKYYT98ooc/mobilebasic)
* [16. POINTERS in C++_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1VJ411M7WR?p=16&vd_source=9c4cffb10e23fffa8fe6d124050c8a48)
* [【27】【Cherno C++】【中字】C++继承_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Yf4y1e7t6/?spm_id_from=333.999.0.0&vd_source=9c4cffb10e23fffa8fe6d124050c8a48)
* [C++ 値渡し、ポインタ渡し、参照渡しを使い分けよう - Qiita](https://qiita.com/agate-pris/items/05948b7d33f3e88b8967)
* [pdf-eigennote](http://ankokudan.org/d/dl/pdf/pdf-eigennote.pdf)
* https://teramonagi.hatenablog.com/entry/20150514/1431633517
* [C++那些事](https://light-city.github.io/)
* [starpos/get-out-of-cpp-beginners: Source code of "Get out of C++ Beginners" book \(in Japanese\)](https://github.com/starpos/get-out-of-cpp-beginners)
* [ゼロから学ぶ C++](https://rinatz.github.io/cpp-book/)
* https://github.com/mirsaidl/C_plusplus
* https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb
* [C++ Cheat Sheet & Quick Reference](https://quickref.me/cpp.html)
* [Skill up with our free tutorials](https://www.learncpp.com/)
* Cmake： [CMake_Cheatsheet](https://usercontent.one/wp/cheatsheet.czutro.ch/wp-content/uploads/2020/09/CMake_Cheatsheet.pdf)
* [E869120/kyopro-tessoku: 拙著『競技プログラミングの鉄則』（2022/9/16 発売）の GitHub ページです。演習問題の解答や、C++ 以外のソースコードなどが掲載されています。ぜひご活用ください。](https://github.com/E869120/kyopro-tessoku/tree/main)
## Google cpp style
* [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
  * Tabs and spaces: 
    * only use spaces and indent 2 spaces at a time. No tabs
  * Type deduction: auto
    * Only to make the code clearer or safer,
    * and do not use it merely to avoid the inconvenience of writing an explicit type.
  * Ownership and smart pointer
    * to prevent memory leak.
    * keep ownership with the code that allocated it,
    * prefer to use std::unique_ptr to make ownership transfer explicit (smart ptr)
  * Exceptions
    * No try catch
  * Iheritance
    * Limit implementation inheritance.
    * Prefer interface inheritance or use **composition instead**.
    * abstract class: no values or code are inherited from the parent.
    * `virtual void speak();`
## Improve cpp abilities
* [[Game Programming]]
* [How to REALLY learn C++](https://www.youtube.com/watch?v=_zQqN5OYCCM)
* https://www.youtube.com/@cppweekly/featured (C++ weekly series)
* [CppCon](https://www.youtube.com/@CppCon)
* [Books-3/Effective C++ 3rd ed.pdf at master · GunterMueller/Books-3](https://github.com/GunterMueller/Books-3/blob/master/Effective%20C%2B%2B%203rd%20ed.pdf)
* [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
## Traning or homeworks
- Atcoder
  - [C++入門 AtCoder Programming Guide for beginners \(APG4b\)](https://atcoder.jp/contests/APG4b)
- [I did a C++ University Assignment](https://www.youtube.com/watch?v=kQsHF7C-FUY&list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb&index=98)
- https://www.reddit.com/r/cpp/comments/r1qr7g/how_to_become_a_better_c_programmer/
  - Try to pick a topic that is in the area of your interests and implement it from A do Z.
  * For me, the golden choice was implementing own game engine
    * (there are a lot of open-source examples so I could compare my ideas with production-ready products).
  * Exploring and understanding standard library
### Tips
- Hashset, counter and vector loop
```cpp
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        unordered_multiset<int> st(nums1.begin(), nums1.end());
        vector<int> ans;
        for (int x: nums2)
        {
            auto it = st.find(x);
            if (it != st.end())
            {
                st.erase(it);
                ans.push_back(x);
            }
        }
        return ans;
    }
};
```
- 2D matrcis + print 2D vectors
```cpp
int main()
{
  int h, w, q, a, b, c, d;
  cin >> h >> w;
  vector<vector<int>> x(h+1, vector<int>(w+1, 0));
  vector<vector<int>> z(h+1, vector<int>(w+1, 0)); 
  for (int i=1; i<=h; i++)
  {
    for (int j=1; j<=w; j++)
    {
      cin >> x[i][j];
    }
  }
// print 2d vectors
  for (int i=1; i<=h; i++)
  {
    for (int j=1; j<=w; j++)
    {
      if (j>1) cout << " ";
      cout << z[i][j];
    }
    cout << endl;
  }
}


set<int> st;
st.insert(C[i]+D[j]);
st.count(k-X[i])

```
- Compertions
```cpp
using namespace std;
using ll = long long;
using ld = long double;
using pl = pair<ll, ll>;

template<class T> using pq = priority_queue<T, vector<T>>;
template<class T> using pq_g = priority_queue<T, vector<T>, greater<T>>;

const ll inf = 2e18;
const ll mod100 = 1e9 + 7;
const ll mod998 = 998244353;
const vector<ll> dx = {0, -1, -1, -1, 0, 1, 1, 1};
const vector<ll> dy = {1, 1, 0, -1, -1, -1, 0, 1};

#define REP(i, n) for (ll i = 0; i < (n); i++)
#define RREP(i, n) for (ll i = (n) - 1; i >= 0; i--)
#define FOR(i, m, n) for (ll i = (m); i < (n); i++)
#define RFOR(i, m, n) for (ll i = (n) - 1; i >= (m); i++)
#define ALL(a) a.begin(), a.end()
#define RALL(a) a.rbegin(), a.rend()
#define NEXT_P(a) next_permutation(ALL(a))
#define UNIQUE(a) a.erase(unique(ALL(a)), a.end())
#define CHMAX(x, y) x = max(x, y)
#define CHMIN(x, y) x = min(x, y)
#define OUT_GRID(x, y, h, w) ((x) < 0 || (x) >= h || (y) < 0 || (y) >= w)
#define E_DIST(x1, y1, x2, y2) (((x1) - (x2)) * ((x1) - (x2)) + ((y1) - (y2)) * ((y1) - (y2)))
#define M_DIST(x1, y1, x2, y2) (abs((x1) - (x2)) + abs((y1) - (y2)))
#define YES cout << "Yes" << endl
#define NO cout << "No" << endl
#define DEBUG(x) cout << #x << ": " << (x) << endl
```
- `std::ios_base::sync_with_stdio`
  - 是 C++ 中一个用于 **同步标准流** 的函数，主要作用是控制 C++ 流（如 cin 和 cout）与 C 标准流（如 stdin 和 stdout）的同步。
  - 其默认行为是 **同步**，即 C++ 的 cin 和 cout 会与 C 的 scanf 和 printf 进行同步操作，以保证两者的输出和输入不会交错。
  - 何时使用：
    * 如果你只使用 **C++ 风格的 I/O**（cin、cout）而不涉及 scanf、printf，可以通过调用 sync_with_stdio(false) 来 **提高性能**。
    * 如果你需要 **频繁进行大量输入输出操作**（如竞赛编程等），这种方式可以显著提高程序运行的效率。
```cpp
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

struct node
{
    int x;
    int id;
    bool operator < (const node& a)const
    {
        return x > a.x;
    }
};

// print a array
int main()
{
    array<int, 10> a = {0};
    for(auto const& value: a)
    {
        std::cout << value << ",";
    }
    return 0;
}
```
- 有限序列 min heap
```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;
const int N = 2e5 + 10;
const int M = (1e4 * 1e3 + 1e4 * 1e3) * 2 + 100;
struct node
{
    int x;
    int id;
    bool operator < (const node& a)const
    {
        return x > a.x;
    }
};

priority_queue<node>q[N];
int ans[N];
int main()
{
    int n, w;
    cin >> n >> w;
    for (int i = 1; i <= n; i++)
    {
        int x, y;
        cin >> x >> y;
        q[x].push({y,i});
    }
    while (1)
    {
        bool f = 0;
        int mx = 0;
        for (int i = 1; i <= w; i++)
        {
            if (q[i].empty())
            {
                f = 1;
                break;
            }
            int x = q[i].top().x;
            mx = max(mx, x);
            // cout << q[i].top().id << ' ';
        }
        if (f) break;
        for (int i = 1; i <= w; i++)
        {
            ans[q[i].top().id] = mx;
            q[i].pop();
        }
    }
    int q;
    cin >> q;
    while (q--)
    {
        int t, x;
        cin >> t >> x;
        if (ans[x] == 0 || ans[x] > t)
            cout << "Yes\n";
        else
            cout << "No\n";
    }
    return 0;
}
```
- [ビット全探索（ 2^n 通りの全探索） | アルゴリズムロジック](https://algo-logic.info/rec-bit-search/)
```cpp
vector<long long> Enumerate(vector<long long> A)
{
  vector<long long> SumList;
  for (int i=0; i< (1<<A.size()); i++)
  {
    long long sum = 0;
    for (int j=0; j < A.size(); j++)
    {
      int wari = (1 << j);
      if ((i/wari)%2 == 1)
      {
        sum += A[j];
      }
    }
    SumList.push_back(sum);
  }

  return SumList;
}
```
- Binary search
```cpp
  sort(Sum1.begin(), Sum1.end());
  sort(Sum2.begin(), Sum2.end());

  for (int i=0; i< Sum1.size(); i++)
  {
    int pos = lower_bound(Sum2.begin(), Sum2.end(), k-Sum1[i]) - Sum2.begin();
    if (pos < Sum2.size() && Sum2[pos] == k-Sum1[i])
    {
      cout << "Yes" << endl;
      return 0;
    }
  }
```
- hash set
```cpp
  sort(t.begin(), t.end());
  t.erase(unique(t.begin(), t.end()), t.end());
  for (int i=0; i < t.size(); i++)
  {
    hm[t[i]] = i+1;
  }

  for (int i=1; i<=n; i++)
  {
    cout << hm[A[i]];
    if (i != n) {
      cout << " ";
    }
  }
```
---
### Head file
* Copy pasted them in current cpp
* bits/stdc++.h
### Pinters
* a int var to save memory address.
* No type, only size. Sizeof them.

### Reference -> Pointer
* More easy to understand.
* `int& a` -> alines
* memory address onto a function

### Class or struct
* class -> More complicated
  * private or public
* struct -> easy, only data and less fucntionals

```cpp
class LOG
{
public:
    const int LogLevelError = 0;
    const int LogLevelWaring = 1;
    const int LogLevelInfo = 2;
private:
    int m_LogLevel = LogLevelInfo;
public:
    void SetLevel(int level)
    {
        m_LogLevel = level;
    }
    void Warn(const char* msg){
        if (m_LogLevel >= LogLevelWaring)
            cout << "[WARING]: " << msg << endl;
    }
    void Error(const char* msg){
        if (m_LogLevel >= LogLevelError)
            cout << "[ERROR]: " << msg << endl;
    }
    void Info(const char* msg){
        if (m_LogLevel >= LogLevelInfo)
            cout << "[INFO]: " << msg << endl;
    }
};
```
### Static in cpp
* Basic by the contex
* static var -> same name var
  * like private
  * externel
  * only be visiable in the cpp file!
  * try to mark them static unless you need them linked!
* Inside a class or struct (P21)
  * change static var will change all val from same entity.
  * Namespace.
  * similar to gloabal variable -> Share data
  * static mthod -> static var -> *static method do not have a instance!*
* Local static
  * where var we can visit.
  * A life time inside the function (local).
  * static var -> gloabal var -> clean code
  * help to init a function
* Enums
  * make code cleaner and easier

```cpp
enum Excample : unsigned char {
    A, B, C
};
// A,B,C = 0, 1, 2
```
### Parts in class
* Constructor
  * same to class name `Entity(){}`
  * run it when it constraction
  * memory init
* Destrctor
  * run when you destroy an object.
  * `~Entity(){}`
* Inheritance
  * To reduce common code
  * basic class -> other functionals -> comman functions
  * Childeren has all the things from father class
  * To apply new functions

```cpp
class Player : public Entity
{
public:
  const char* Name;
  void PrintName()
		std::cout << Name << std::endl;
}
```
* Virtual function
  * Rewrite functions
  * Dynamic Dispatch -> v table -> mapping to virtual functions -> override functions
  * Not free. 
    * need memory for v table and pointers, go through all tables.
    * That is OK~

```cpp
// father class
virtual std::String GetName() {};

// childeren class
std::String GetName() override {};
```
* pure vitural function -> interface class
  * necessary to be rewritten in childeren class.
  * must need a childeren.

```cpp
// father class
virtual std::String GetName() = 0;

// childeren class
std::String GetName() override {};
```
* Feasibility
  * for better code and style -> nothing todo with performance
  * private -> inside class and friends
  * protected -> only in class and sub class
  * public -> everyone

### Array, string and vars
* Array -> ponters -> A little bit faster
  * A collection for vars
  * `int exp[5];` `exp[0] = 23;` -> in a stack
    * `int count = sizeof(exp) / sizeof(int);`
  * Memory acces violation -> care about it!
    * `for (int i =0; i < 5; i+=) {exp[i] = 2; };`
  * `index+ptr`
  * `new` -> will not delete -> memory indirection -> low speed (in a heap)
  * `std::array` in C++ 11
    * bound checking, record size of array, more safe.

```cpp
#inclued <array>

int* ptr = exp;
exp[2] == *(ptr+2) // move 2*4 (int ptr)

int* another = new int[5];
delete[] antoher

std::array<int, 5> exp;
exp.size(); 
```
* String
  * char -> 1B -> Ascii -> UTF -> 16B
  * String -> Array of chars
  * `const char* name = "example";` -> `""` means char pointer
    * `const` -> fixed memory block -> Cannot change 
    * no `new`, no `delete`.
  * `\0` or `0` -> end mark
  * `std::string` 
    * size()
    * strlen()
    * strcopy()
    * `name += " hello!";`
    * `boll contains = name.find("no") != std::string::npos;`
    * string copy -> too slow
  * string var

```cpp
const char* name = u8"xx";
const wchar_t* name = L"xx";
const char16_t* name = u"xx"; //
const char32_t* name = U"xx";

using namespace std::string_literals; // cpp 14
std::string name = "AAA"s + "BBB";

const char* exp = R"(
aSD
asd
)";
```
* Const
  * to make code clean
  * const首先作用于左边的东西；如果左边没东西，就做用于右边的东西
  * `const int* a = new int`
    * 表示a指向的地址中的内容不能被修改，可以理解为我把指向内容的int类型声明为const不准修改了，因为const和int在一起
  * `int* const a = new int`
    * 表示a指向的地址不能修改，可以理解为这个const和指针名在一起，表示a存储的值（指向的地址）不能改变
  * In a class, after a method name.
    * This method cannot modifiy any vars in the class.
    * I donot want tocopy some class, I need refer.
      * I donot want to modify my class or instance.
    * to use the const method n other functions
    * Always mark your function which donot modify class.
    * Use `mutable` for you var, when you want to modify them in `const` method.

```cpp
const int* a = new int;
*a = 2; // NG
a = (int*)&MAX_AGE; // OK But not so good
std::cout << *a << std::end;

public:
		int GetX() const
			{}

void fun(const Entity& e) // you can call any const functions
		{
			e.GetX()
		}
```
* Mutable
  * Use with const
    * Use `mutable` for you var, when you want to modify them in `const` method.
  * lambdas -> less

```cpp
int x=8;
auto f = [=]() mutable
{
		x++ ; 
		std::cout << x << std::endl; // x = 9
};

f(); // x = 8
```
* Constrctor member init list
  * var init
    * 顺序一致
  * Why?
    * code style -> more clean
    * no need to init 2 times -> More fast
  * Use it everywhere.

```cpp
std::string m_Namel;
Entity()
		: m_Name("Unknown"), score(0)
{}
Entity(const std::string& name)
		: m_Name(name)
{}
```
* c++三元运算符
  * Sugar for `if`
  * Make code clean

```cpp
using String = std::string;
a = s > 5 ? 10 : 5;
```
### Constract and init cpp object
* Where it goes? stack or heap
  * stack -> life span -> small (1~2 M) -> in `{}` -> Faster
  * heap -> Large object or need to be delete -> `new` -> Memory leak (smart pointers)

```cpp
Entity* e = new Entity("name");
(*e).GetName(); // e->GetName()
delete e;
```
* `new` key word -> operator
  * init on stack -> memory -> free list
  * return a pointer
  * memory (`malloc(sizeof(Entity))`) + init class

```cpp
int a = 2;
int* b = new int[50]; // 200 B
Entity* e = new Entity();

delete e;
delete[] b;
```
* implicit and explicit
  * automatic
  * 隐式变换只能做一次 -> simply code -> Not recommendation
  * explicit (constractor) -> disable implicit -> No auto cast
* operator and overloaded -> Bad
  * a sample to insterad of a function
  * not recommendation -> dangerous
* this -> pinter to current object
  * to visit class member function
* Object life span for stack
  * between `{}` -> scope
    * in `{}` donot create array bu stack
      * ``
  * scoped_lock
  * scoped pointer
* smart pointer
  * new -> malloc memory, delete -> release memory (doonot need to call it)
  * no `new` and `delete` in cpp
  * unique_ptr
    * scope pointer -> out of scope -> auto delete
    * cannot copy -> only reference
    * `std::make_unique<Entity>()` -> safety (nullptr -> memory leak)
  * shared_ptr
    * reference counting in std -> how many reference count to your pointer
    * reference count == 0 -> delete
  * weak_ptr
    * reference count cannot be increase!
    * not keep it alive
  * Try to use them all the time.
    * cannot replace new and delete

```cpp
#inclued <memory>

class Entity
{
public:
		Entity()
		{
			std::cout << "Created" << std::endl;
		}
		~Entity()
		{
			std::cout << "Destroyed" << std::endl;
		}
}

int main()
{
	{
		std::shared_ptr<Entity> e0；
		{
			std::unique_ptr<Entity> entity = std::make_unique<Entity>(); 
			//new Entity() is ok, but not safe.
			std::shared_ptr<Entity> sharedEntity = std::make_shared<Entity>();
			e0 = sharedEntity;
			std::weak_ptr<Entity> wp = sharedEntity;

			entity->Print();
		}
	}	
}
```
* Copy 拷贝构造函数
  * copy need time -> to avoid
  * alwyas copy the value
    * use `new` or copy a pointer -> not copy -> memory address (reference)
    * on stack -> copy 
  * deep copy -> copy a object with new memory
    * copy constractor -> copy 
  * shallow copy -> int of ptr 
  * **Always pass your object by const refernce!**
    * No reason to copy your data when you use a function.
    * You can decied to copy in your function.

```cpp
class String()
{
private:
	char* m_b;
	unsigned int m_size;
public:
	String(const char* string)
	{
		m_szie = strlen(string);
		m_b = new char[m_size];
		memcpy(m_b, string, m_size);
		m_b[m_size] = 0;
	}

	~String()
	{
		delete[] m_b;
	}

	char& operator[](unsigned int index)
	{
		return m_b[index];	
	}

/*
	String(const String& other)
	{
		memcpy(this, &other, sizeof(String));
	}
*/
	String(const String& other) //deep copy
		:m_size(other.m_size)
	{
		m_b = new char[m_size+1];
		memcpy(m_b, &other.m_b, m_size+1);
	}

	friend std::oststream& operator<< (std::ostream& stram, const String& string);
};

std::oststream& operator<< (std::ostream& stram, const String& string)
{
	stream << string.m_b;
	return stream;
}

void PrintString(const String& string) // add `&` would not copy when print
{
	String copy = string; // You need copy?
	std::cout << string << std::endl;
}

int main()
{
	String string = "CSADAsasd";
	String copyString = string; // copy ptr or object
	
	PrintString(string);
	
}

```
* Arrow operator -> shortcut
  * method or vars
  * overloaded

```cpp
class ScopedPtr()
{
public:
	ScopedPtr(Entity* enity)
		: m_obj(entity)
	{}
	
	~ScopedPtr()
	{delete m_obj;}

	Entity* operator->()
	{
		return m_obj;
	}
	const Entity* operator->() const
	{
		return m_obj;
	}
}

struct Vector3
{
    float x, y, z;
}

int main()
{
	Entity e;
	e.Print();

	Entity* ptr = &e;
	(*ptr).Print();
	ptr->Print();

	const ScopePtr entity = new Entity*();
	entity->Print();

  int offset = (int)&((Vector3*)nullptr)->z; 
  std::cout << offset << std::endl;
  std::cin.get();
  return 0;
}
```
* std::vector -> ArryList
  * auto resize
    * copy and resize
  * reserve提前申请内存，避免动态申请开销
  * emplace_back直接在容器尾部创建元素，省略拷贝或移动过程

```cpp
struct Vertex
{
    float x, y, z;
};

std::ostream& operator<<(std::ostream& stream, const Vertex& vertex)
{
    stream << vertex.x << "," << vertex.y << "," << vertex.z;
    return stream;
}

int ArrayList()
{
    std::vector<Vertex> vertices; // ArrayList
/* slow more copy
    vertices.push_back({1, 2, 3});
    vertices.push_back({4, 5, 6});
*/
		vertices.reserve(3);
		vertices.emplace_back(1, 2, 3);
		vertices.emplace_back(4, 5, 6);
		vertices.emplace_back(7, 8, 9);

    for (int i=0; i < vertices.size(); i++)
        LOG(vertices[i]);

    vertices.erase(vertices.begin()+1);

    for (auto& v: vertices) // & to prevent copy
        LOG(v);

    vertices.clear();
}
```
* Library in cpp, Static link (P49) -> more optimization
  * Binary -> quick and easy
  * include and library -> linker
  * mkdir `dependencies/GLFS` -> put download files there 
    * `.dll` and `.lib` -> dynamic link
    * `*dll.lib`  -> static link
  * project property -> Configuration -> Add folder path (Additional include direactions) (Macros)
    * linker -> genreal -> Addtional include direactions
  * `#include "Folder/name.h"`
    * if `.h` outside solution , use `<>` -> external
    * `extern "C" int glfwInit();` outside -> C and Cpp
* Dynamically lib
  * head file is same -> add file name 
  * copy dll with exe folder
* Create lib
  * inclued lib `.h` file
* Return multi values, tuple, pair
  * A function return 1 type.
  * 如果函数需要返回多个返回值，还是使用结构体…
  * tuple
    * `#include <utility>` `#include <tuple>`
    * `std::tuple<std::string, std::string>`
    * `return std::make_pair(v1, v2);`
    * `std::get<0>(shader)`
    * `shader.first`
  * Structer -> more clear
    * `return {a, b};`
  * **Structure bindings -> cpp17**
    * return values -> return structs
    * make our code more cleaner
    * `std::tuple` `std::pair` `std::tie`

```cpp
std::tuple<std::string, int> CreatePerson()
{
    return {"somebody", 28};
}

void StructBindingStudy()
{
    auto[name, age] = CreatePerson();
    LOG(name);
    LOG(age);
}
```
* Template -> Macro
  * blue-print -> code generation
  * only template, constract when we use them
  * meta programming
  * Understand is important

```cpp
template<typename T> // template name
void Print(T value)
{
    std::cout << value << std::endl;
}

template<typename T, int N>
class Array
{
private:
    T m_Array[N];
public:
    int GetSize() const {return N;}
};

int main() {
    ArrayList();
    Print<int>(1);
    Print("!23");

    Array<std::string, 5> array;
    LOG(array.GetSize());

    return 0;
}
```
* memory between stack and heap -> allocation method is different
  * stack -> 2M -> fast -> use it -> long life span
  * heap -> can grow up -> malloc -> return pointer -> cache miss -> slow
  * free list
* cpp Macro
  * use preprocessor to speed up and automation
  * `#define WAIT std::cin.get()` search and replace `WAIT;`
  * preprocess parameters -> Debug or Release

```cpp
#define PR_DEBUG = 0
#if 0 // Fold code
// #ifdef PR_DEBUG
#if PR_DEBUG == 1 // use \ to next line
#define xxxx
#elif defined(PR_DEBUG)
#define xxxxx
#endif
```
* auto
  * need type?
  * for loop iterator
  * A large and long type `using devicetype = std::...;` or use `auto`
* std::array -> stantic array
  * bound check, fast, opt

```cpp
template<int N>
void PrintArray(std::array<int, N>& arr)
{
 for (int i = 0; i < arr.size(); i++)
 {
  std::cout << arr[i] << std::endl;
 }
}
```
* function ptr
  * function as var

```cpp
void Hello()
{
    LOG("Hello");
}

void Hello(int a)
{
    LOG(a);
}

void Foreach(const std::vector<int>& values, void(*func)(int))
{
    for (int value : values)
        func(value);
}

int main()
{
    typedef void(*HelloFunction)(int);

    auto function = Hello;
    HelloFunction fun = Hello;

    function(3);
    fun(43);

    std::vector<int> values = {12,3,4,5};
    Foreach(values, fun);
    Foreach(values, [](int val){LOG(val);});
    return 0;
}
```
* lambda 匿名函数
  * if you have a fucntion ptr, youvan use lambda
  * [] -> capture -> put outside var into lambda function
  * [Lambda expressions (since C++11) - cppreference.com](https://en.cppreference.com/w/cpp/language/lambda)

```cpp
void Foreach(const std::vector<int>& values, void(*func)(int))
{
    for (int value : values)
        func(value);
}

int main()
{
    typedef void(*HelloFunction)(int);
	   int a = 5;
    auto lambda = [=](int value) {std::cout << "Value: " << value << std::endl;}; //[&a]

    auto function = Hello;
    HelloFunction fun = Hello;

    function(3);
    fun(43);

    std::vector<int> values = {12,3,4,5};
    Foreach(values, fun);
    Foreach(values, [](int val){LOG(val);});
    Foreach(values, lambda);

    return 0;
}
```
* Why donot use namespace std? (60)
  * 其他非STL库里可能有同名函数 然后会有静默运行错误，很难追踪bug
* Name space in cpp
  * `using namespace std` -> in a fucntion -> Not in top or head file
  * `namespace apple {}` -> Same name question -> comman name is NG
  * nest namespace is ok
* cpp threads
  * `#include <tread>`

```cpp
void DoWork()
{
    using namespace std::chrono;

    std::cout << "Start thread id = " << std::this_thread::get_id() << std::endl;
    while (!s_Finished)
    {
        std::cout << "Working...\n";
        std::this_thread::sleep_for(minutes(1));
    }

}

void ThreadStudy()
{
    std::thread worker(DoWork);

    std::cin.get();
    s_Finished = true;

    worker.join(); // wait or wait for end
    std::cin.get();
}
```
* timing std::chrono
  * `#include chrono`

```cpp
#include <chrono>
struct Timer
{
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<float> duration;
    Timer()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        end = std::chrono::high_resolution_clock::now();
        duration = end-start;
        float ms = duration.count() * 1000.0f;
        std::cout << "Timer took " << ms << "ms" << std::endl;
    }
};

void TimeFunction()
{
    Timer timer;
    for (int i = 0; i < 100; i++)
        LOG("Hello");
}

void TimeStudy()
{
    TimeFunction();
}
```
* cpp multi-dim array
  * array -> block of memory -> ptr
  * nest loop for int -> memory allocate -> A cube
  * cache miss -> slow
  * use 1 array to insdead mulit-dim array

```cpp
void MultiDimArray()
{
    int** a2d = new int*[50];
    for (int i = 0; i < 50; i++)
    {
        a2d[i] = new int[50];
    }

    a2d[0][0] = 0;

    for (int i = 0; i < 50; i++)
        delete[] a2d[i];
    delete[] a2d;

		int* array = new int[5*5];
		for (int y = 0; y < 5; y++)
    		for (int x = 0; x < 5; x++)
        		array[x+y*5] = 2; // Faster
}
```
* Sort in cpp -> std::sort
  * https://en.cppreference.com/w/cpp/algorithm/sort
  * `std::sort(values.begin(), values.end(), std::greater<int>());`

```cpp
void SortStudy()
{
    std::vector<int> values = {1, 2, 4, 6, 3, 9};
//    std::sort(values.begin(), values.end(), std::greater<int>());
    std::sort(values.begin(), values.end(), [](int a, int b)
    {
        if (a==1)
            return false;
        if (b == 1)
            return true;
    });

    for (auto value : values)
        LOG(value);
}
```
* type punning
  * you can acess memeory directly. -> fast 
  * avoid type in cpp
  * we got a memory and we can treat them as any types. -> type of ptr
* union
  * union里的成员会共享内存，分配的大小是按最大成员的sizeof
  * 视频里有两个成员，也就是那两个结构体，改变其中一个另外一个里面对应的也会改变. 如果是这两个成员是结构体`struct{int a,b}`和`int k`, 如果k=2 ; 对应 a也=2 ，b不变； 
  * union我觉得在这种情况下很好用，就是用不同的结构表示同样的数据 ，那么你可以按照获取和修改他们的方式来定义你的 union结构

```cpp
struct Vector2
{
    float x, y;
};

struct Vector4
{
    union
    {
        struct
        {
            float x, y, z, w;
        };
        struct
        {
            Vector2 a, b;
        };
    };
};

void PrintVector2(const Vector2& vector){
    std::cout << vector.x << ",　" << vector.y << std::endl;
}

void UnionStudy()
{
    struct Union
    {
        union
        {
            float a;
            int b;

        };
    };
    Union u;
    u.a = 2.0f;
    std::cout << u.a << ",　" << u.b << std::endl;

    Vector4 vector = {1.0f, 2.0f, 3.0f, 4.0f};
    PrintVector2(vector.a);
    PrintVector2(vector.b);
    vector.z = 500.f;
    LOG("----------------------------------");
    PrintVector2(vector.a);
    PrintVector2(vector.b);
}
```
* Virtual constractor
  * 如果用基类指针来引用派生类对象，那么基类的析构函数必须是 virtual 的，否则 C++ 只会调用基类的析构函数，不会调用派生类的析构函数。
  * Father class must be `virtual ~ClassName()` -> memory leak

```cpp
class Base
{
public:
    Base()
    {
        std::cout << "Constructed\n";
    }
    virtual ~Base()
    {
        std::cout << "Destructed\n";
    }
};

class Derived : public Base
{
public:
    Derived()
    {
        std::cout << "Derived Constructed\n";
    }
    ~Derived()
    {
        std::cout << "Derived Destructed\n";
    }
};

void VirtualConsStudy()
{
    Base* base = new Base();
    delete base;
    LOG("---------------------");
    Derived* derived = new Derived();
    delete derived;
    LOG("---------------------");
    Base* poly = new Derived(); //memory leak
    delete poly;
}
```
* Casting in cpp
  * Type system
  * cast types
    * `static_cast`  `dynamic_cast` `reinterpret_cast` `const_cast`
  * `static_cast` 
    * 用于进行比较“自然”和低风险的转换，
    * 如整型和浮点型、字符型之间的互相转换,
    * 不能用于指针类型的强制转换
  * `reinterpret_cast` 
    * 用于进行各种不同类型的指针之间强制转换
  * `const_cast` -> 仅用于进行去除 const 属性的转换
  * `dynamic_cast` -> 不检查转换安全性，仅运行时检查，如果不能转换，返回null

```cpp
void CastStudy()
{
    double value = 5.25;
    double a = (int)value+5; // c style
    double s = static_cast<int>(value)+5;
    LOG(s);

    Derived* derived = new Derived();
    Base* base = derived;

    Derived* ac = dynamic_cast<Derived*>(base);
    if (ac)
    {
        LOG("OK");
    }
}
```
* break points -> conditions and actions
  * debug -> condition or actions -> `{(float)var}. {(float)y}` -> No break
  * Speed up our work
* 用于生产环境使用智能指针，用于学习和了解工作积累，使用原始指针
  * 永远用智能指针
* Precomplied header
  * share code, such as vector or head files, can be precomplied as binary format file, to reduce complied time.
  * Donot put offen changed files into your head files-> slow.
  * `#include "pch.h"` -> contians all stl headFiles `#pragam once`
  * set on software gcc etc…
* Dyanmic_cast in cpp
  * Like a function to ertify cast is OK or not.
  * inhertence class cast check
  * check type -> RTTI
* Benchmark
  * Compare speed of code
  * timer in a scope
  * `__debugbreak()`
  * Visualy test
    * cpp的计时器配合自制简易json配置写出类，将时间分析结果写入一个json文件，用chrome://tracing 这个工具进行可视化
    * [Basic Instrumentation Profiler · GitHub](https://gist.github.com/TheCherno/31f135eea6ee729ab5f26a6908eb3a5e)
    * [GitHub - GavinSun0921/InstrumentorTimer: 利用chrome://tracing实现的C++的可视化基准测试](https://github.com/GavinSun0921/InstrumentorTimer)
* Optinal data -> cpp17
  * `#include <optional>`
  * `std::optional<type> function(param){statement; return type;}`
  * `auto result = function();`
  * 1: `result.has_value()`判断数据是否存在, 通过`result.value()`获取数据
  * 2: `result.value_or(xxx)`其中xxx作为默认值，如果存在数据返回数据，不存在返回xxx
  * 3:通过`if (result)`判断数据是否存在
  * 注: 使用场景—目标值可能存在也可能不存在，比如读取文件并返回内容，可能读取成功有数据，读取成功无数据，读取不成功。
* How to store multi type data in a var -> cpp 17 -> a class
  * `#include <variant>` -> similar to optional
    * 类似于union，type1与type2表示存储的数据类型。
    * give detail infromation of error
    * type safe

```cpp
std::variant<type1, type2> data;
data = type1(xxx) // size = sizeof(type1)+sizeof(type2)
// load data
std::get<type>(data)
auto *value = std::get_if(type)(&data)
```
* 如何存储任意类型的数据 -> cpp17
  * `#include <any>` `std::any`
  * `std::string = std::any_cast<std::string>(data);`
  * dynanic memory

### Make cpp more fast -> MP
* `#include <future>`
* mutex -> proces lock `std::mutex`
* 为什么不能传引用？
  * 线程函数的参数按值移动或复制。
  * 如果引用参数需要传递给线程函数，它必须被包装（例如使用std :: ref或std :: cref）
* `std::async` -> cpp 11
  * 为什么一定要返回值？ `std::future`
    * 如果没有返回值，那么在一次for循环之后，临时对象会被析构，而析构函数中需要等待线程结束，所以就和顺序执行一样，一个个的等下去
    * 如果将返回值赋值给外部变量，那么生存期就在for循环之外，那么对象不会被析构，也就不需要等待线程结束。
* Make cpp string more fast
  * allocate memory -> stack or heap
  * `string_view` -> cpp17
    * a ptr and a size -> substr() -> no memory allocations -> fast
    * `std::string_view firstName(name.c_str(), 3);`
    * use char array
  * sso -> small string opt -> not on heap
    * static string -> small then length -> vs2019 is 15 char
    * 
* Singleton -> only one instance
  * sometimes. we did not need many objects. only one.
  * == global vars and static functions -> single namespace
    * no constructor in public
    * static function in public

```cpp
class Random
{
public:
    Random(const Random &) = delete;
    static Random& Get()
    {
        static Random s_Instance;
        return s_Instance;
    }
    static float Float() { return Get().IFloat(); }
private:
    Random() {}
    float m_RandomGenerator = 0.5f;
    float IFloat(){return m_RandomGenerator;}
};

void SingletonStudy()
{
    float number = Random::Float();
}
```
* Memory allocate -> how to detect heap allocation
  * vector size
  * memory arena
  * 

```cpp
struct AllocationMetrics
{
    uint32_t TotalAllocated = 0;
    uint32_t TotalFreed = 0;

    uint32_t CurrentUsage() {return TotalAllocated-TotalFreed; }
};
static AllocationMetrics s_AllocationMetrics;

void* operator new(size_t size)
{
    s_AllocationMetrics.TotalAllocated += size;
    return ::malloc(size);
}

void operator delete(void* memory, size_t size)
{
    s_AllocationMetrics.TotalFreed += size;
    std::free(memory);
}

static void PrintMemoryUsage()
{
    std::cout << "Memory Usage: " << s_AllocationMetrics.CurrentUsage() << " bytes\n";
}
```
* l and r values
  * [video](https://www.bilibili.com/video/BV1Aq4y1t73p/?spm_id_from=pageDriver&vd_source=9c4cffb10e23fffa8fe6d124050c8a48)
  * l-val -> located val -> address -> mostly on the left of = -> reference
  * r-val -> mostly on the right of = -> no reference
  * `void SetValue(const int& values)` -> support l or r value
  * `void SetValue(std::string& values)` -> only l value -> save val
  * `void SetValue(std::string&& values)` -> only r value -> detection -> temp val
* static analysis -> clang-tidy
  * improve code -> check your code
* argument evaluation order 
  * 因为参数的evaluated顺序不确定，所以不要重载&&，||和逗号操作符
  * `printsum(val++, val++);` -> undefined behaviour -> compiler depanded
* move in cpp
  * `std::move` -> move an object into antoher object
  * https://github.com/UrsoCN/NotesofCherno/blob/main/Cherno89-90.cpp

```cpp
    String(String &&other) // 移动构造函数
    {
        printf("Moved!\n");
        m_Size = other.m_Size;
        m_Data = other.m_Data;
        other.m_Data = nullptr;
        other.m_Size = 0; // 偷掉原来的String
    }
apple = std::move(orange)
```
### Array
```
template<typename T, size_t S>
class Array
{
public:
    constexpr int Size() const {return S;}
    T& operator[](int index)
    {
        if (index < S)
        {
            return m_Data[index];
        }
    }
    const T& operator[](int index) const {return m_Data[index];}
    T* Data() {return m_Data; }
    const T* Data() const {return m_Data; }
private:
    T m_Data[S];
};
```
### Vector
* [自己动手写Vector【Cherno C++教程】 - zhangyi1357 - 博客园](https://www.cnblogs.com/zhangyi1357/p/16009968.html)
* No use it !
- - -
## Refers
* [GitHub - PacktPublishing/Hands-On-Machine-Learning-with-CPP: Hands-On Machine Learning with C++, published by Packt](https://github.com/PacktPublishing/Hands-On-Machine-Learning-with-CPP)
* [YouTube](https://www.youtube.com/watch?v=E1K9SZCm0fQ&list=PL79n_WS-sPHKklEvOLiM1K94oJBsGnz71)
* [C++入門](http://wisdom.sakura.ne.jp/programming/cpp/)
* http://dlib.net/ml.html
* [Top C/C++ Machine Learning Libraries For Data Science | HackerNoon](https://hackernoon.com/top-cc-machine-learning-libraries-for-data-science-nl183wo1)
## OpenGL
- [AndroidNote/VideoDevelopment/OpenGL/1.OpenGL简介.md at master · CharonChui/AndroidNote](https://github.com/CharonChui/AndroidNote/blob/master/VideoDevelopment/OpenGL/1.OpenGL%E7%AE%80%E4%BB%8B.md)
### Welcome to OpenGL
- Graphic API
  - to control GPU
- Specifitvation (cpp) of what you can do with the API
  - No code, just spectification
- GPU —> drivers —> contained the implementation of open GL
- You cannot see the code for OpenGL
- Cross-platform
- 