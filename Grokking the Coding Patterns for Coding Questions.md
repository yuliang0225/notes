#  Grokking the Coding: Patterns for Coding Questions
#study/interview 
[[数据结构]]
## Lists
* [[Leetcode question list]]
* [LeetCode按照怎样的顺序来刷题比较好？ - 知乎](https://www.zhihu.com/question/36738189/answer/797854274)
* https://leetcode.cn/circle/discuss/RvFUtj/
* [codeforces-go/leetcode/SOLUTIONS.md at master · EndlessCheng/codeforces-go](https://github.com/EndlessCheng/codeforces-go/blob/master/leetcode/SOLUTIONS.md)
* [OI Wiki](https://oi-wiki.org/)
## Refers
* [刷题进阶Tips-分享给那些有刷题经验或工作经验的人|一亩三分地刷题版](https://www.1point3acres.com/bbs/thread-289223-1-1.html)
* [コーディング面接対策のために解きたいLeetCode 60問 | 新井康平](https://1kohei1.com/leetcode/)
* [https://github.com/liyin2015/Algorithms-and-Coding-Interviews](https://github.com/liyin2015/Algorithms-and-Coding-Interviews) 
* [https://1kohei1.com/google/](https://1kohei1.com/google/) 
* http://kaiyuzheng.me/dump/notes/interview.pdf
* [GitHub - VincentUCLA/LCPython](https://github.com/VincentUCLA/LCPython)
* [GitHub - SeanPrashad/leetcode-patterns: A curated list of 160+ leetcode questions grouped by their common patterns](https://github.com/SeanPrashad/leetcode-patterns)
* [GitHub - Dharni0607/Leetcode-Questions: Leetcode question list by companies, includes the premium questions. December 2019 updated](https://github.com/Dharni0607/Leetcode-Questions)
* [educative-io-contents/Grokking Dynamic Programming Patterns for Coding Interviews.md at master · asutosh97/educative-io-contents · GitHub](https://github.com/asutosh97/educative-io-contents/blob/master/Grokking%20Dynamic%20Programming%20Patterns%20for%20Coding%20Interviews.md)
* [Grokking Dynamic Programming Patterns for Coding Interviews - Learn Interactively](https://www.educative.io/courses/grokking-dynamic-programming-patterns-for-coding-interviews?coupon_code=dp-1point3acres&affiliate_id=5749180081373184/)
* [关于课程”Grokking Dynamic Programming Patterns for Coding Interviews”|一亩三分地公开课版](https://www.1point3acres.com/bbs/thread-503954-1-1.html)
* [GitHub - ShusenTang/LeetCode: LeetCode solutions with Chinese explanation & Summary of classic algorithms.](https://github.com/ShusenTang/LeetCode)
* [Introduction - coding practice - advanced topics](https://po-jen-lai.gitbook.io/coding-practice-advanced-topics/)
* [Grokking the Coding Interview: Patterns for Coding Questions - Learn Interactively](https://www.educative.io/courses/grokking-the-coding-interview)
* [chxj1992/leetcode-concurrency-programming: LeetCode Concurrency Programming Exercise](https://github.com/chxj1992/leetcode-concurrency-programming)
- - -
## How To Pass Coding Interviews Like the Top 1%
- https://www.youtube.com/watch?v=fQW6-2yfsBY
- Plan, write and explain your code
- Planing
  - Clarify the problem
    - code —> input and output
    - how to communicate and do understand the problem
      - ask about 3 different edges cases to prove that you are a problem solver
    - input size, with negative numbers or anything else when you’re done with the probelm
    - **Hated: Forces everyone to think about the optimal perfect solution every single time.**
      - We should start with a naive solution. (dump, simple solution) —> Start point
      - Care about
        - your thought process —> simple test cases + different algorithms
        - Finally with the best solution —> Improve
    - Write down your algorithm (outline) = your thought
      - Funcation of the next coding interview
      - Communicate with your interviewers
      - Speak out loud and ask the interviewer if they have any questions.
  - Explain the algorithm
- Writing
  - What you write down is exactly what you mentioned in your outline
  - Evaluation: Communication and Coding efficiency.
  - Do not be silent, we should communicate with interviewers, why, what, how with speaking.
  - Talk about the code, how ite relates to the outline that you wrote earlier.
    - what you’re going to write and how it relates to your algorithm
    - and then code in silence
      - input and what it will be returned
  - good vars name, how easy it is to read your code
  - Stuck part
    - react to it determines wheather or not you get an offer
    - cannot stuck on a small coding issue, Todo: later, skip it. write others
    - // Todo: null checks
- Explaning
  - Ask the interviewer todo:
  - dry run your code
    - prove your code works
    - picking a test case and run it in your code line by line
    - edge casese —> how the input slowly becomes the expected output
    - input size matters. —> small test cases
  - optimize the solution
    - talk about the bottleneck in you code
      - Each session your code and writing down runing time O(1), O(N), O(NlogN)
      - Point it out
    - optimize your code and explaining thought process all over again —> repeat
---
## Tree basic
* [力扣](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/solution/python3-die-dai-bian-li-chang-gui-jie-fa-xnim/)
* Heap https://en.wikipedia.org/wiki/Binary_heap
  * Insert: Average O(1)
  * Extract: log(N)
  * Search: N
  * Delete: logN
* [递归和迭代的区别 - 知乎](https://zhuanlan.zhihu.com/p/49600594#:~:text=%E9%80%92%E5%BD%92%E6%98%AF%E9%87%8D%E5%A4%8D%E8%B0%83%E7%94%A8%E5%87%BD%E6%95%B0,%E5%BE%AA%E7%8E%AF%E8%AE%A1%E7%AE%97%E7%9A%84%E5%88%9D%E5%A7%8B%E5%80%BC%E3%80%82&text=%E9%80%92%E5%BD%92%E5%BE%AA%E7%8E%AF%E4%B8%AD%EF%BC%8C%E9%81%87%E5%88%B0,%E9%80%90%E5%B1%82%E8%BF%94%E5%9B%9E%E6%9D%A5%E7%BB%93%E6%9D%9F%E3%80%82)
  * 1、程序结构不同
    * 递归是重复调用函数自身实现循环。迭代是函数内某段代码实现循环。 其中，迭代与普通循环的区别是：
    * 迭代时，循环代码中参与运算的变量同时是保存结果的变量，当前保存的结果作为下一次循环计算的初始值。
  * 2、算法结束方式不同
    * 递归循环中，遇到满足终止条件的情况时逐层返回来结束。
    * 迭代则使用计数器结束循环。 
    * 当然很多情况都是多种循环混合采用，这要根据具体需求。
  * 3、效率不同
    * 在循环的次数较大的时候，迭代的效率明显高于递归。
* BFS+DFS
  ![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/1166454-20200727175140915-644646597.png)
## 如何科学刷题？
- https://leetcode.cn/circle/discuss/RvFUtj/
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/image%202.png)<!-- {"width":605} -->
- **训练方法 A**
  - 按照**专题**刷题。同一个专题下的题目，套路是一样的，刷题效率杠杠滴~
  * **从易到难**，量力而行。题目已经按照难度分整理好了，按照顺序刷就行
  * 对于**动态规划**，至少要做100 道才算入门。
  * 优点
    * 突击训练特定知识点，可以掌握常用算法套路。
    * 按照题单专题刷，一个套路可以解决多个题目，刷题效率高。
    * 此外，做同一个专题下的题目，相当于在从不同的角度去观察、思考同一个算法，这能让你更加深刻地理解算法的本质。
  * 缺点
    * 提前知道题目类型，跳过了一些思考步骤。
    * 但比赛/笔试是不会告诉你这题是什么类型的，把 DP 想成贪心的大有人在。
    * 可以结合下面的训练方法，锻炼自己判断题目类型的能力。
* **训练方法 B**
  * 随机刷题，注意**控制难度**范围，太简单和太难的题目都不能让你进入「心流通道」。
  * [LC-Rating & Training](https://huxulm.github.io/lc-rating/zen)
    * 1 在设置中**关闭算法标签**。
    * 2 选择适合自己的难度范围，开刷！
  * **优点**做题时不知道题目类型，可以训练实战能力。
  * **缺点**：知识点有些零散，适合已经掌握常用算法的同学查漏补缺，检验自己的学习成果。
* **训练方法 C**
  * https://leetcode.cn/studyplan/top-100-liked/

- - -
## 1. Pattern: Sliding window
* Array + Hash tables -> Memory
[Subarrays with K Different Integers - LeetCode](https://leetcode.com/problems/subarrays-with-k-different-integers/discuss/523136/JavaC%2B%2BPython-Sliding-Window)
[Sliding Window - LeetCode](https://leetcode.com/tag/sliding-window/)
```python
start, max_len = 0,0
dict = {}
for end in range(len(s)):
	if s[end] not in dict:
		dict[s[end]] = 0
	dict[s[end]] += 1
	while … :
		left = s[start]
		start += 1
		dict[left] -= 1
		if dict[left] == 0:
			del dict[left]
	max_len = max(max_len, end - start + 1)
	…
```
* 5/4 5/18 5/27 5/28 5/29 6/12 6/25 7/1 7/25 9/12 9/16 9/21 9/22
* 2/18
* **2024**
  * 4/13  4/14  5/25 6/1
  * 7/7 9/10 9/11
* **2025**
  * 2/5
```python
# basic
**-*-**--*---****209. Minimum Size Subarray Sum
******-****159.Longest Substring with At Most Two Distinct Characters
******-******904. Fruit Into Baskets
****-*****340. Longest Substring with At Most K Distinct Characters
*********3. Longest Substring Without Repeating Characters
-*-**-*-*------**-**76. Minimum Window Substring #
***-*--****53. Maximum Subarray
*****1800. Maximum Ascending Subarray Sum

# median
***-**------*--**-**424. Longest Repeating Character Replacement
**-**--*--**-------**-*567. Permutation in String
****-**-*--*-***438. Find All Anagrams in a String
--*--*--*------*--325. Maximum Size Subarray Sum Equals k

# hard
---67. Add Binary
--30. Substring with Concatenation of All Words
472. Concatenated Words
727. Minimum Window Subsequence
395. Longest Substring with At Least K Repeating Characters
992. Subarrays with K Different Integers
108. Convert Sorted Array to Binary Search Tree
242. Valid Anagram

2516. Take K of Each Character From Left and Right
3258. Count Substrings That Satisfy K-Constraint I
```
- - -
## 2 Pattern: Merge Intervals 
* Array + Sort
* 7/23 7/25 
* 5/13 5/14 5/15 5/30 6/14 6/27 6/28 9/28
* 2/18 2/19
* **2024**
  * 4/13 7/6 7/8 9/11 9/12
* **2025**
  * 2/4 
```python
Sort by start
If first.end >= second.start:
	merge

Maybe use heap to store end time
```

```python
********-*****56. Merge Intervals
*****-*******252. Meeting Rooms
*--*-***-----*--**253. Meeting Rooms II
**-*-*----*****57. Insert Interval
--**-**-**-----*-**986. Interval List Intersections
-*---*****759. Employee Free Time #
```
- - -
## 3 Pattern: Cyclic Sort 
* Array
  * This pattern describes an interesting approach to deal with problems **involving arrays containing numbers in a given range**. 
* To efficiently solve this problem, we can use the fact that the input array contains numbers in the range of 1 to ‘n’. 
  * For example, to efficiently **sort the array**, we can try placing each number in its correct place, i.e., placing ‘1’ at index ‘0’, placing ‘2’ at index ‘1’, and so on. 
  * Once we are done with the sorting, we can iterate the array to find all indices that are **missing the correct numbers**. 
  * These will be our required numbers.
* Let’s jump on to our first problem to understand the **Cyclic Sort** pattern in detail.
* Sort method
  * [Data Structure Visualization](https://www.cs.usfca.edu/~galles/visualization/Algorithms.html)
  * heap rank
    * Time (NlogN) Space (N)
  * Quick sort: l + [pivot] + r Add random
    * Time (NlogN) Space (logN)
```python
Sort by swap
if nums[I] != nums[nums[i]-1]: swap
Else i += 1
OR 

while i < n:
	j = nums[i]
	if nums[i] < n and nums[i] != nums[j]:
		swap
	else i +=1

	return first nums[i] != i
```
* 7/24 5/15 5/31 6/14 6/28 6/29 
* 9/29
- **2024**
  - 4/13 4/14 7/9 7/19 9/12 9/13
- **2025**
  - 2/3 2/4
```python
***-*****912. Sort an Array
**-*-*--*--*-*-**268. Missing Number
**--*----*-*-*--287. Find the Duplicate Number 
**----*---****-*442. Find All Duplicates in an Array
-----*----**--*41. First Missing Positive

--*-***-*1539. Kth Missing Positive Number
-*-*-*------*148. Sort List
*---*---*147. Insertion Sort List
----*--*1060. Missing Element in Sorted Array 
```
### **【套路】教你解决定长滑窗！适用于所有定长滑窗题目！**
- https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/solutions/2809359/tao-lu-jiao-ni-jie-jue-ding-chang-hua-ch-fzfo/
- https://leetcode.cn/circle/discuss/0viNMK/
- 入-更新-出。
  - 入：下标为 i 的元素进入窗口，更新相关统计量。如果 i<k−1 则重复第一步。
  - 更新：更新答案。一般是更新最大值/最小值。
  - 出：下标为 i−k+1 的元素离开窗口，更新相关统计量。

- - -
## 4. Pattern: Two Points 
* Array + Binary serach
[Two Pointers - LeetCode](https://leetcode.com/tag/two-pointers/)
* Given an array of **sorted** numbers and a target sum,
  * find a pair in the array whose sum is equal to the given target.
* Consider Hash table to make a memory.
* 注意去重 如何处理重复元素问题
```python
left, right = 0, len(a)-1
while left < right:
	if a[left] + a[right] > target:
		right -= 1
	elif a[left] + a[right] < target:
		left += 1
	else:
		return [left, right]
```
* 4/20 7/19 7/24 5/6 5/29 6/12 6/13 6/14 6/25 6/26 7/1
* 9/21 9/23
* **2024**
  * 4/13 4/14 7/20 9/14 9/15
* **2025**
  * 2/2 2/3
```python
# 2 pointer sum question
****-*******1. Two Sum
**********167. Two Sum II - Input array is sorted
*-***-*-**--*653. Two Sum IV - Input is a BST
**---*-*-*--16. 3Sum Closest
--**--*-*-*---*-*-**-*15. 3Sum
*-***-----***18. 4Sum

# remove resort question
***-*--*27. Remove Element
**-****-**-*283. Move Zeroes
*--*-*----*-----------*---75. Sort Colors
**--****-------*-**26. Remove Duplicates from Sorted Array
**----**---------80.Remove Duplicates from Sorted Array II

# others
***********977. Squares of a Sorted Array
****-**844. Backspace String Compare
--------*-----581.Shortest Unsorted Continuous Subarray
*----**---38. Count and Say
---------713. Subarray Product Less Than K
*--*-*-----*560. Subarray Sum Equals K
```
- - -
## 5 Pattern: Modified Binary Search
* Array + Binary Search
* As we know, whenever we are given a sorted **Array** or **LinkedList** or **Matrix**, and we are asked to find a certain element, the best algorithm we can use is the  [Binary Search](https://en.wikipedia.org/wiki/Binary_search_algorithm) .
* This pattern describes an efficient way to handle all problems involving **Binary Search**. 
  * We will go through a set of problems that will help us build an understanding of this pattern so that we can apply this technique to other problems we might come across in the interviews.
* [二分查找的坑点与总结_haolexiao的专栏-CSDN博客](https://blog.csdn.net/haolexiao/article/details/53541837)
  * 以下这个函数是二分查找nums中[left，right)部分，right值取不到，如果查到的话，返回所在地，如果查不到则返回最后一个小于target值得后一个位置。
```cpp
//右值点不能取到的情况
    int binary_search(vector<int>& nums,int left,int right, int target) { 
//坑点（1）right究竟能不能取到的问题，这里是不能取到的情况
        int i = left;
        int j = right;
        while(i<j){
            int mid = i+(j-i)/2;             
//坑点（2）这里尽量这么写，因为如果写成(i+j)/2则有溢出的风险
            if(nums[mid]>=target)        
//坑点（3）这个地方大于还是大于等于要依据情况而定
                j = mid;            
//坑点（4）因为右值点反正不能取到，所以j就可以等于mid
            else
                i = mid+1;           
//坑点（5）依据right能不能取到而定，如果right可以取到则，right必须要-1，不减1的话，还是会出现i = j时的死循环。
        }
        return i;
    }

//右值点能取到的情况
    int searchInsert(vector<int>& nums,int left,int right, int target) {
        int i = left;
        int j= right;
        while(i<=j ){
            int mid = i+(j-i)/2;
            if(nums[mid]>=target)
                j = mid-1;
            else
                i = mid+1;
        }
        return i;
    }
```

```python
start, end = 0, len(nums)-1
while start <= end:
	mid = start + (end-start)//2
	if nums[mid] == target:
		return mid
 	if target in [start, mid]:
		end = mid - 1
	else:
		start = mid + 1
```
* 4/29 5/20 5/21 5/22 6/5 6/22 6/23 7/3 7/4 7/12
* 10/13 10/14 10/16
* **2024**
  * 4/14 7/26 8/1 9/16 9/17
* **2025**
  * 2/1 2/3
```python
# Part 1
**********704. Binary Search
***-*-*-*---*-*744. Find Smallest Letter Greater Than Target
***-*-*-******1150. Check If a Number Is Majority Element in a Sorted Array
****--*-*-*--****34. Find First and Last Position of Element in Sorted Array
***-*---*-*-----**--162. Find Peak Element

# Part 2
*---**74. Search a 2D Matrix
----**-*----*-*----1095. Find in Mountain Array
*--*-*--*--*---*--*33. Search in Rotated Sorted Array

*-**---*-*--------81. Search in Rotated Sorted Array II
--*-*--*--*---**--153. Find Minimum in Rotated Sorted Array
--------*---*-154. Find Minimum in Rotated Sorted Array II

# Part 3
-*-540. Single Element in a Sorted Array
---702. Search in a Sorted Array of Unknown Size
---1283. Find the Smallest Divisor Given a Threshold
```
- - -
## 6 Pattern: Fast & Slow pointers 
* Linked list
* The Fast & Slow pointer approach, also known as the Hare & Tortoise algorithm, is a pointer algorithm that uses two pointers which move through the array (or sequence/LinkedList) at different speeds. 
* This approach is quite useful when dealing with cyclic LinkedLists or arrays.
```python
slow, fast = head, head
slow = slow.next
fast = fast.next.next
if slow == fast: ---
while fast is not None and fast.next is not None: ---
cyc_len += 1
start cycle: 
	fast = slow + cyc_len; slow == fast
```
* 4/18 7/23 7/24 5/8 5/30 6/12 6/14 6/27 6/28 9/28
* 2/18 2/19
- **2024**
  - 04/14 04/15 8/7 8/8 9/18
- **2025**
  - 1/31 2/1
```python
*****-*-*-****141. Linked List Cycle
**-**-********876. Middle of the Linked List
*-*-***-**--*---*---202. Happy Number
--***---***--*---*-***142. Linked List Cycle II
--*-**-*--*-*---*-*234. Palindrome Linked List

--*---*----*160. Intersection of Two Linked Lists
--*--**---------*--143. Reorder List
----------------*--457. Circular Array Loop
-*--*----*208. Implement Trie (Prefix Tree)
```
- - -
## 7 Pattern: In-place Reversal of a LinkedList
* Linked list
* In a lot of problems, we are asked to *reverse* the links between a set of nodes of a LinkedList. 
  * Often, the constraint is that we need to do this in-place, i.e., using the existing node objects and without using extra memory.
  * In-place Reversal of a LinkedList pattern describes an efficient way to solve the above problem. 
* 5/16 5/17 5/18 6/16 6/18 6/19 6/30 
* 9/29 2/19 2/20
* **2024** 
  * 4/15 4/16 8/11 8/12 9/19 9/20 9/21
* **2025**
  * 1/30 1/31 2/1
``` python
reveser 
cur, prev = head, None
while cur:
	tmp = cur.next
	cur.next = prev
	prev = cur
	cur = tmp
```

``` python
**********206. Reverse Linked List
******---*83. Remove Duplicates from Sorted List
---*---*-*-------*--*92. Reverse Linked List II
--*---*-*----*--*--*24. Swap Nodes in Pairs
--*-----*-*-------**25. Reverse Nodes in k-Group

--*----*------*-*-*61. Rotate List
-----*-----*82. Remove Duplicates from Sorted List II
--***--*-**19. Remove Nth Node From End of List
***-*-***203. Remove Linked List Elements
**----**160. Intersection of Two Linked Lists
***------*328. Odd Even Linked List
```
- - -
## 8 Pattern: Tree Breadth First Search BFS
* Tree
* Any problem involving the traversal of a tree in a level-by-level order can be efficiently solved using this approach. 
* We will use a Queue to keep track of all the nodes of a level before we jump onto the next level. 
* This also means that the space complexity of the algorithm will be O(W), where ‘W’ is the maximum number of nodes on any level.
  * Time: N Space: N
```python
use collections.deque
while queue:
levelSize and currentLevel
for _ in range(levelSize)
	currentNode = queue.popleft()
	currentLevel.append(currentNode.val)
	if currentNode.left:
		queue.append()
	if ….right:
		queue.append
result.append(cuurentLevel)
```
* In order Successor
  * need parent part for each nodes
```python
        if node.right:
            node = node.right
            while node:
                s_p = node
                node = node.left
            return s_p
        else:
            while node.parent and node.parent.right == node:
                node = node.parent            
            return node.parent
```
* 4/25 5/9 6/1 6/19 6/30 
* 10/6 10/7 10/28
* **2024**
  * 04/16 9/21
* **2025**
  * 1/29 2/4
```python
# Part 1
******---****102. Binary Tree Level Order Traversal
******107. Binary Tree Level Order Traversal II
******---***103. Binary Tree Zigzag Level Order Traversal
***-****637. Average of Levels in Binary Tree
*******515. Find Largest Value in Each Tree Row

# Part 2
******---***104. Maximum Depth of Binary Tree
****-*-***116. Populating Next Right Pointers in Each Node
*-***117. Populating Next Right Pointers in Each Node II
****-***429. N-ary Tree Level Order Traversal
***-****199. Binary Tree Right Side View

# Part 3
-**--*-----*111. Minimum Depth of Binary Tree
---**-----*---285. Inorder Successor in BST
**-*---*---*510. Inorder Successor in BST II
--*-----*863. All Nodes Distance K in Binary Tree
*540. Single Element in a Sorted Array
```
- - -
## 9 Pattern: Subsets BFS
* Tree
* A huge number of coding interview problems involve dealing with  [Permutations](https://en.wikipedia.org/wiki/Permutation)  and  [Combinations](https://en.wikipedia.org/wiki/Combination)  of a given set of elements. 
  * This pattern describes an efficient **Breadth First Search (BFS)** approach to handle all these problems.
  * Time and space: O(N∗2^N)
```python
def find_subsets(nums):
  subsets = []
  # start by adding the empty subset
  subsets.append([])
  for currentNumber in nums:
    # we will take all existing subsets and insert the current number in them to create new subsets
    n = len(subsets)
    for i in range(n):
      # create a new subset from the existing subset and insert the current element to it
      set1 = list(subsets[i])
      set1.append(currentNumber)
      subsets.append(set1)

  return subsets
```
* 4/26 5/19 5/20 6/3 6/4 6/5 6/21 6/22 7/2 7/3 7/4 7/12
* 10/7 10/8 10/10 10/28
- **2024**
  - 04/17 9/21 9/22 9/23
- 2025
  - 1/28 2/4
```python
# Part 1
**-****-**--*-**---**-*-*78. Subsets
******--*-*-*-*--**90. Subsets II
*****--**---**--46. Permutations
-47. Permutations II
*****--*-*---*-**22. Generate Parentheses
***----*----*---*-*320. Generalized Abbreviation

# Part 2
****-*--*-*----*-*241. Different Ways to Add Parentheses

--***--*-*---*-784. Letter Case Permutation
------*------*-96. Unique Binary Search Trees
-------*-*----*-95. Unique Binary Search Trees II
---*-297. Serialize and Deserialize Binary Tree
```
- - -
## 10 Pattern: Tree Depth First Search DFS
* Tree
* DFS approach
* This also means that the space complexity of the algorithm will be O(H), where ‘H’ is the maximum height of the tree.
```python
def fun(self, root, target):
		if not root:
			return Fasle
		if target == root.val and Blablabla:
			return True 
		return self.fun(root.left, target-root.val) or self.fun(root.right, target-root.val)
```
1. We will keep track of the current path in a list which will be passed to every **recursive call**.
2. Whenever we traverse a node we will do two things:
   * Add the current node to the current path.
   * As we added a new node to the current path, we should find the sums of all sub-paths ending at the current node. 
   * If the sum of any sub-path is equal to ‘S’ we will increment our path count.
3. We will traverse **all paths** and will not stop processing after finding the first path.
4. **Remove** the current node from the current path before returning from the function. 
   * This is needed to **Backtrack** while we are going up the recursive call stack to process other paths.
* 4/23 4/25 5/10 5/12 5/13 6/2 6/3 6/19 6/20 7/1
* 10/7
* **2024**
  * 04/18 04/20 9/23 9/24 9/25 9/26 9/27
* **2025**
  * 1/24 1/26 1/27
```python
# Part 1
***-***-**---***112. Path Sum
***--*-----***-113. Path Sum II
*****-***257. Binary Tree Paths
***-*----**-200. Number of Islands
***---*-----*-1522. Diameter of N-Ary Tree

# Part 2
***-*-**589. N-ary Tree Preorder Traversal
******590. N-ary Tree Postorder Traversal
***---**--*-*---*-*-39. Combination Sum
-**----**-------*-129. Sum Root to Leaf Numbers
-*---*-988. Smallest String Starting From Leaf
-*40. Combination Sum II

# Part 3
-****--**872. Leaf-Similar Trees
--*--*---**---------*-437. Path Sum III
-*--*-**-----*-543. Diameter of Binary Tree
------*-----*-687. Longest Univalue Path
-*-*-*-------*-124. Binary Tree Maximum Path Sum
-*--**--*-366. Find Leaves of Binary Tree
*-**216. Combination Sum III
```
- - -
## 11 Pattern: Two Heaps
* Heaps
* In many problems, where we are given a set of elements such that we can divide them into **two parts**. 
  * To solve the problem, we are interested in knowing
    * the smallest element in one part 
    * and the biggest element in the other part. 
  * This pattern is an efficient approach to solve such problems.
* This pattern uses two **Heaps** to solve these problems; 
  * A **Min Heap** to find the smallest element
  * and a **Max Heap** to find the biggest element.
* [Python：从堆中删除元素 - 程序园](http://www.voidcn.com/article/p-bdykbbrr-bsy.html)
* [Python 的 heapq 模块源码分析 | 码农网](https://www.codercto.com/a/49843.html)
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/image.png)<!-- {"width":467} -->
```python
from heapq import *

# add nums
maxHeap, minHeap = [], []
if not maxHeap or -maxHeap[0] >= num:
	heappush(maxHeap, -num)
else:
	heappush(minHeap, num)

if len(maxHeap) - len(minHeap) > 1:
	heappush(minHeap, -heappop(maxHeap))
elif len(minHeap) > len(maxHeap):
	heappush(maxHeap, -heappop(minHeap))

# return median
if len(maxHeap) == len(minHeap):
	median = -maxHeap[0]/2.0 + minHeap[0]/2.0
else:
	median = -maxHeap[0]/1.0
```

```python
# remove or add new element need rebalance
  # removes an element from the heap keeping the heap property
  def remove(self, heap, element):
    ind = heap.index(element)  # find the element
    # move the element to the end and delete it
    heap[ind] = heap[-1]
    del heap[-1]
    # we can use heapify to readjust the elements but that would be O(N),
    # instead, we will adjust only one element which will O(logN)
    if ind < len(heap):
      heapq._siftup(heap, ind)
      heapq._siftdown(heap, 0, ind)

  def rebalance_heaps(self):
    # either both the heaps will have equal number of elements or max-heap will have
    # one more element than the min-heap
    if len(self.maxHeap) > len(self.minHeap) + 1:
      heappush(self.minHeap, -heappop(self.maxHeap))
    elif len(self.maxHeap) < len(self.minHeap):
      heappush(self.maxHeap, -heappop(self.minHeap))
```
* 4/23 5/18 6/3 6/21 7/2 7/12 
* 10/17 10/18
* **2024**
  * 04/19 04/20 5/7 5/8 5/9 9/28 9/29
* **2025**
  * 1/20 1/22
```python
*-***-*--*----*****295. Find Median from Data Stream
---*--*-*----**-*480. Sliding Window Median
--*--*-*-------**502. IPO
----*-----*-436. Find Right Interval
```
- - -
## 12 Pattern: Top ‘K’ Elements
* Heaps + Hash tables
* Any problem that asks us to find the top/smallest/frequent ‘K’ elements among a given set falls under this pattern.
  * The best data structure that comes to mind to keep track of ‘K’ elements is **[Heap](https://en.wikipedia.org/wiki/Heap_(data_structure))**. 
  * This pattern will make use of the Heap to solve multiple problems dealing with ‘K’ elements at a time from a set of given elements.
``` python
The best data structure that comes to mind to keep track of top ‘K’ elements is Heap. 

Considered Hashtable combo with heap

Time: NLogK
Space: K
```
* 4/30 5/22 5/23 5/24 6/6 6/7 7/5 7/10
* 11/2
- **2024**
  - 04/21 5/6 9/29 9/30 
- **2025**
  - 1/19 
```python
# Part 1
---****-*****215. Kth Largest Element in an Array
****--****973. K Closest Points to Origin
****--***-1167. Minimum Cost to Connect Sticks
****--***-347. Top K Frequent Elements
*******451. Sort Characters By Frequency

# Part 2
*******703. Kth Largest Element in a Stream
*--*-*--*-----*-658. Find K Closest Elements
*****1133. Largest Unique Number
**-*-*-*--*-*-767. Reorganize String
--*-------358. Rearrange String k Distance Apart

# Part 3
-*----621. Task Scheduler
*-*---895. Maximum Frequency Stack
--*---692. Top K Frequent Words
```
- - -
## 13 Pattern: K-way merge
* Heap + Binary search
* This pattern helps us solve problems that involve a list of sorted arrays.
  * Whenever we are given ‘K’ sorted arrays, we can use a **Heap** to efficiently perform a sorted traversal of all the elements of all arrays. 
  * We can push the smallest (first) element of each sorted array in a **Min Heap** to get the overall minimum. 
  * While inserting elements to the **Min Heap** we keep track of which array the element came from. 
  * We can, then, remove the top element from the heap to get the smallest element and push the next element from the same array, to which this smallest element belonged, to the heap. 
  * We can repeat this process to make a sorted traversal of all elements.
```python
from heapq import *
class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        if not nums:
            return None
        l, r = 0, float("inf")
        min_heap = list()
        cur_max = -float("inf")

        for num in nums:
            cur_max = max(cur_max, num[0])
            heappush(min_heap, (num[0], 0, num))
        
        while len(min_heap) == len(nums):
            cur_min, ind, arr = heappop(min_heap)
            if cur_max-cur_min < r-l:
                l, r = cur_min, cur_max
            if ind+1 < len(arr):
                cur_max = max(cur_max, arr[ind+1])
                heappush(min_heap, (arr[ind+1], ind+1, arr))
        
        return [l, r]
```
* 5/4 5/23 5/24 6/7 7/6
* **2024**
  - 04/21 04/28 5/5 5/6 10/1
- **2025**
case [Stats dimension]
when 'Has Children' then [Has Children]
when 'Has Property' then [Has Property]
when 'Gender' then [Gender]
when 'Is Married' then [Is Married]
when 'Is Married' then [Is Married]
else [Mkt Channel] END

- 1/18 2/23
```python
*-***-*-*-**23. Merge k Sorted Lists
*********21. Merge Two Sorted Lists
**-*--*-***378. Kth Smallest Element in a Sorted Matrix
---**----*----632. Smallest Range Covering Elements from K Lists
---**-----*---373. Find K Pairs with Smallest Sums

4. Median of Two Sorted Arrays
```
- - -
## 14 Pattern: Bitwise XOR
* Bitwise
* XOR is a logical bitwise operator that returns 0 (false) if both bits are the same and returns 1 (true) otherwise. 
  * In other words, it only returns 1 if exactly one bit is set to 1 out of the two bits in comparison.
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2021-05-22%2011.22.21.png)
* Double numbers
  * Find groups
  * 1 XOR 0 = 0 XOR 1 = 1
  * rightmostSetBit =1 >> 1
* 从集合论到位运算，常见位运算技巧分类总结 https://leetcode.cn/circle/discuss/CaOJ45/
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2025-01-18%2017.59.31.png)<!-- {"width":594.88636363636363} -->
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2025-01-18%2017.59.55.png)<!-- {"width":594} -->
* 4/29 4/30 5/22 6/6 7/3 7/12
- 2024
  - 5/4 5/5 5/6
- **2025**
  - 1/18
``` python
^ same => 0
^ not same => 1
```
- Basic list
```python
******-*136. Single Number
-*----*-137. Single Number II
---**-----260. Single Number III
-*-*----*-1009. Complement of Base 10 Integer
***-----*832. Flipping an Image
```
- 基础题
  - https://leetcode.cn/circle/discuss/dHn9Vk/
```python

```

- - -
## 15 Pattern: Topological Sort (Graph) 
* Graph
* [Topological Sort](https://en.wikipedia.org/wiki/Topological_sorting)  is used to find a linear ordering of elements that have dependencies on each other. 
  * For example, if event ‘B’ is dependent on event ‘A’, ‘A’ comes before ‘B’ in topological ordering.
  * This pattern defines an easy way to understand the technique for performing topological sorting of a set of elements and then solves a few problems using it.
```python
# init graph
	inDegree = {i: 0 for i in range(tasks)}  
# count of incoming edges
	graph = {i: [] for i in range(tasks)}  
# adjacency list graph

# build graph
  for prerequisite in prerequisites:
    parent, child = prerequisite[0], prerequisite[1]
    graph[parent].append(child)  
# put the child into it's parent's list
    inDegree[child] += 1  
# increment child's inDegree

# find sources
  sources = deque()
  for key in inDegree:
    if inDegree[key] == 0:
      sources.append(key)

# d. For each source, add it to the sortedOrder and 
# subtract one from all of its children's in-degrees
# if a child's in-degree becomes zero, add it to the # sources queue
  while sources:
    vertex = sources.popleft()
    sortedOrder.append(vertex)
    for child in graph[vertex]:  
# get the node's children to decrement 
# their in-degrees
      inDegree[child] -= 1
      if inDegree[child] == 0:
        sources.append(child)

# test
  if len(sortedOrder) != tasks:
    return []

  return sortedOrder
```
* Time: O(V+E)
* Space: O(V+E)
* **2022~2023**
  - 5/4 5/5 6/9 7/7 9/30 10/1 10/3 10/6
- **2024**
  - 5/5 5/6
- **2025**
  - 1/18 2/11 2/23
```python
****--**207.Course Schedule
**--*-*-**210.Course Schedule II
-----*--*-*---269.Alien Dictionary
*-*-*--***444.Sequence Reconstruction
*----*----310.Minimum Height Trees
```
- https://leetcode.cn/circle/discuss/01LUak/
- - -
# DP Based
## 16 Pattern: Fibonacci numbers
* DP
* Fibonacci number pattern
  * `CountWays(n) = CountWays(n-1) + CountWays(n-2) + CountWays(n-3) + … + CountWays(n-k), for n >= k`
  * `dp[end] = Math.min(dp[end], dp[start]+1);`
```python
def count_min_jumps(jumps):
  n = len(jumps)
  # initialize with infinity, except the first index which should be zero as we
  # start from there
  dp = [math.inf for _ in range(n)]
  dp[0] = 0

  for start in range(n - 1):
    end = start + 1
    while end <= start + jumps[start] and end < n:
      dp[end] = min(dp[end], dp[start] + 1)
      end += 1

  return dp[n - 1]
```
* 7/14 
* 9/29 9/30 10/1 10/6 10/10 10/23
* 10/26 10/27 10/28
* **2024**
  * 5/11 5/12 6/29
- **2025**
  - 1/13 1/14 1/17 2/9 2/10 2/19 
```python
# Part 1
***-*******509. Fibonacci Number
***-****-***70. Climbing Stairs
**-*-*---**-*-*-**55. Jump Game
***---**-*-**198. House Robber
*--*------*--------**746. Min Cost Climbing Stairs

# Part 2
*--**--*-***213. House Robber II
**-*-*-***53. Maximum Subarray
-*------**-----*45. Jump Game II
*-------*-*-*983. Minimum Cost For Tickets

# Part 3
***303. Range Sum Query - Immutable
----*-368. Largest Divisible Subset
----**338. Counting Bits
----**264. Ugly Number II
-----*673. Number of Longest Increasing Subsequence

-309. Best Time to Buy and Sell Stock with Cooldown
------337. House Robber III


2209. Minimum White Tiles After Covering With Carpets
```
- - -
## 17 Pattern : 0/1 Knapsack
* DP
* **0/1 Knapsack** pattern is based on the famous problem with the same name which is efficiently solved using **Dynamic Programming (DP)**.
  * In this pattern, we will go through a set of problems to develop an understanding of DP. 
  * We will always start with a brute-force recursive solution to see the overlapping subproblems, i.e., realizing that we are solving the same problems repeatedly.
  * After the recursive solution, we will modify our algorithm to apply advanced techniques of **Memoization** and **Bottom-Up Dynamic Programming** to develop a complete understanding of this pattern.
* 0/1 背包问题：
  * 最基本的背包问题就是 0/1 背包问题：
    * 一共有 N 件物品，第 i（i 从 1 开始）件物品的重量为 w[i]，价值为 v[i]。
    * 在总重量不超过背包承载上限 W 的情况下，能够装入背包的最大价值是多少？
* 完全背包问题：
  * 完全背包与 0/1 背包不同就是每种物品可以有无限多个：
    * 一共有 N 种物品，每种物品有无限多个，第 i（i 从 1 开始）种物品的重量为 w[i]，价值为 v[i]。
    * 在总重量不超过背包承载上限 W 的情况下，能够装入背包的最大价值是多少？
    * 可见 0/1 背包问题与完全背包问题主要区别就是**物品是否可以重复选取**。
* 背包问题具备的特征：
  * 是否可以根据一个 **target**（直接给出或间接求出），target 可以是数字也可以是字符串，再给定一个数组 arrs，问：能否使用 arrs 中的元素做各种**排列组合**得到 target?
* 背包问题解法：
  * 0/1 背包问题：
    * 如果是 0/1 背包，即数组中的元素**不可重复使用**，外循环遍历 arrs，**内循环遍历 target，且内循环倒序**:
  * 完全背包问题：
    * 1. 如果是完全背包，即数组中的元素可重复使用并且不考虑元素之间顺序，arrs 放在外循环（*保证 arrs 按顺序*），**target在内循环。且内循环正序**。
    * 2. 如果组合问题需考虑元素之间的顺序，**需将 target 放在外循环，将 arrs 放在内循环，且内循环正序**。
``` python
s = sum(nums)
if target > s or (s+target)%2 != 0:
    return 0
s = (s+target)//2
n = len(nums)
dp = [1] + [0] * (s)

for i in range(n):
    for j in range(s, nums[i]-1, -1):
        dp[j] = dp[j] + dp[j-nums[i]]

return dp[-1]
```
* 5/4 5/23 5/24 6/8 7/6 7/7 7/9 7/12 7/13 7/23 9/30 
* 10/1 10/5 10/6 10/10 10/23 10/27 10/28 10/29 10/30
* **2024**
  * 5/12 5/26 5/28 6/30 7/3
* **2025**
  * 1/13 1/14 1/16 1/17 1/19 2/8 2/18 2/20
```python
# Part 1
***---**--*-*---*---**-**-*39. Combination Sum
****-*---*----****322. Coin Change
*****---*---**-***518. Coin Change 2
**--*--*-*--*--**--*-***279. Perfect Squares
**--------**-*221. Maximal Square

# Part 2
**------*-*---*-*-*--139. Word Break
*---*-*-*---*-------**-494. Target Sum
*-*----**----*--**-*-*-*416. Partition Equal Subset Sum
*-*--------*---***377. Combination Sum IV
---*-----------1235. Maximum Profit in Job Scheduling
*-----------2008. Maximum Earnings From Taxi

# Part 3
--2218. Maximum Value of K Coins From Piles
```
[LintCode77：Longest Common Subsequence_网络_卷卷萌的博客-CSDN博客](https://blog.csdn.net/mengmengdajuanjuan/article/details/85372596)
- - -
## 18 Pattern: Palindromic Subsequence
* DP
* Given a sequence, find the length of its Longest Palindromic Subsequence (LPS). 
* In a palindromic subsequence, elements read the same backward and forward.
* Manacher’s Algorithm
  ` Minimum_deletions_to_make_palindrome = Length(st) - LPS(st)`
* **Basic Solution**
  * A basic brute-force solution could be to try all the subsequences of the given sequence. 
  * We can start processing from the beginning and the end of the sequence.
    * So at any step, we have two options:
    1. If the element at the beginning and the end are the same, we increment our count by two and make a recursive call for the remaining sequence.
    2. We will skip the element either from the beginning or the end to make two recursive calls for the remaining subsequence.
  * If option one applies then it will give us the length of LPS;
    * otherwise, the length of LPS will be the maximum number returned by the two recurse calls from the second option.
  * In each function call, we are either having one recursive call or two recursive calls (we will never have three recursive calls); 
    * hence, the time complexity of the above algorithm is exponential O(2^n), where ‘n’ is the length of the input sequence. 
    * The space complexity is O(n), which is used to store the recursion stack.
```python
def find_LPS_length(st):
  return find_LPS_length_recursive(st, 0, len(st) - 1)

def find_LPS_length_recursive(st, startIndex, endIndex):
  if startIndex > endIndex:
    return 0
  # every sequence with one element is a palindrome of length 1
  if startIndex == endIndex:
    return 1
  # case 1: elements at the beginning and the end are the same
  if st[startIndex] == st[endIndex]:
    return 2 + find_LPS_length_recursive(st, startIndex + 1, endIndex - 1)
  # case 2: skip one element either from the beginning or the end
  c1 = find_LPS_length_recursive(st, startIndex + 1, endIndex)
  c2 = find_LPS_length_recursive(st, startIndex, endIndex - 1)
  return max(c1, c2)
```
* **Bottom-up Dynamic Programming**
  * Since we want to try all the subsequences of the given sequence,
    * we can use a **two-dimensional** array to store our results.
  * We can start from the beginning of the sequence and keep adding one element at a time. 
  * At every step, we will try all of its subsequences. 
  * So for every startIndex and endIndex in the given string, we will choose one of the following two options:
    1. If the element at the startIndex matches the element at the endIndex, the length of LPS would be two plus the length of LPS till startIndex+1 and endIndex-1.
    2. If the element at the startIndex does not match the element at the endIndex, we will take the maximum LPS created by either skipping element at the startIndex or the endIndex.
  * So our recursive formula would be:
```python
if st[endIndex] == st[startIndex] 
  dp[startIndex][endIndex] = 2 + dp[startIndex + 1][endIndex - 1]
else 
  dp[startIndex][endIndex] = Math.max(dp[startIndex + 1][endIndex], dp[startIndex][endIndex - 1])
```
* The time and space complexity of the above algorithm is O(n^2), where ‘n’ is the length of the input sequence.
```python
def find_LPS_length(st):
  n = len(st)
  # dp[i][j] stores the length of LPS from index 'i' to index 'j'
  dp = [[0 for _ in range(n)] for _ in range(n)]

  # every sequence with one element is a palindrome of length 1
  for i in range(n):
    dp[i][i] = 1

  for startIndex in range(n - 1, -1, -1):
    for endIndex in range(startIndex + 1, n):
      # case 1: elements at the beginning and the end are the same
      if st[startIndex] == st[endIndex]:
        dp[startIndex][endIndex] = 2 + dp[startIndex + 1][endIndex - 1]
      else:  # case 2: skip one element either from the beginning or the end
        dp[startIndex][endIndex] = max(
          dp[startIndex + 1][endIndex], dp[startIndex][endIndex - 1])

  return dp[0][n - 1]
```
* 7/14 7/16 7/18 7/22 10/2 10/4 10/5
* 10/23 10/30 11/1 11/2
* **2024**
  * 5/23 5/25 5/26
  * 6/30 7/1 7/3
* **2025**
  * 1/15 1/16 1/18 1/19 2/6 2/7 2/17 2/18
```python
# Part 1
****--*-******409. Longest Palindrome
*--*-----*--****--*516. Longest Palindromic Subsequence
*-***-*--*------*5. Longest Palindromic Substring
*-****----*******1312. Minimum Insertion Steps to Make a String Palindrome
*-**-****--*1400. Construct K Palindrome Strings

# Part 2
*-------*-*---*-*-*-*647. Palindromic Substrings
*------------*-*-*131. Palindrome Partitioning
--------------*----132. Palindrome Partitioning II
*----------*---1653. Minimum Deletions to Make String Balanced
*--------*---*---1682. Longest Palindromic Subsequence II

-------------730. Count Different Palindromic Subsequences
```
- - -
## 19 Pattern: Longest Common Substring
* Given two strings ‘s1’ and ‘s2’, 
  * find **the length of the longest substring** which is common in both the strings.
* **Maximum sum increasing subsequence (MSIS)**
  * `if num[I] > num[j] => dp[I] = dp[j] + num[I] if there is no bigger MSIS for ‘I’`
* **Shortest common super-sequence (SCS)**
```python
if s1[i] == s2[j] 
  dp[i][j] = 1 + dp[i-1][j-1]
else 
  dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1]) 
```
* Completely match ‘m’ and ‘n’ (the two interleaving strings) with ‘p
```python
dp[mIndex][nIndex] = false
if m[mIndex] == p[pIndex] 
  dp[mIndex][nIndex] = dp[mIndex-1][nIndex]
if n[nIndex] == p[pIndex] 
 dp[mIndex][nIndex] |= dp[mIndex][nIndex-1]
```
- LCS
```python
def find_LCS_length(s1, s2):
  n1, n2 = len(s1), len(s2)
  dp = [[0 for _ in range(n2+1)] for _ in range(n1+1)]
  maxLength = 0
  for i in range(1, n1+1):
    for j in range(1, n2+1):
      if s1[i - 1] == s2[j - 1]:
        dp[i][j] = 1 + dp[i - 1][j - 1]
      else:
        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

      maxLength = max(maxLength, dp[i][j])
  return maxLength
```
* 7/22 7/24 10/3 10/4 10/5 10/20 10/21
* 10/30
* **2024**
  * 5/24 5/25 5/26
  * 7/1 7/2 7/3
  * 11/12
* **2025**
  * 1/16 1/18 1/19 2/6 2/7 2/16 2/17
```python
# Part 1
**-*-**--*******1143. Longest Common Subsequence
******-*******583. Delete Operation for Two Strings
**-*------------*-*-*---1062. Longest Repeating Substring
**-*---*--**-**-*--*72. Edit Distance
**-*-*----*-*-*-*--*97. Interleaving String
-*--**-*-----*-*--*---*300. Longest Increasing Subsequence
--*--*----*-*-*-**--*1092. Shortest Common Supersequence
```
## 20 Pattern: Segment tree
- 
- [8.8.3.线段树（无区间更新） - 力扣（LeetCode）全球极客挚爱的技术成长平台](https://leetcode.cn/problem-list/cAa6H7Yn/)
- [8.8.4.Lazy 线段树（有区间更新） - 力扣（LeetCode）全球极客挚爱的技术成长平台](https://leetcode.cn/problem-list/zmUOSV0x/)
- **2025**
  - 2/22
- 
```python
# 线段树 无区间更新
1157. Online Majority Element In Subarray
2407. Longest Increasing Subsequence II
2940. Find Building Where Alice and Bob Can Meet
2286. Booking Concert Tickets in Groups
2213. Longest Substring of One Repeating Character

# Lazy 线段树 有区间更新
2589. Minimum Time to Complete All Tasks
2569. Handling Sum Queries After Update
1622. Fancy Sequence
2916. Subarrays Distinct Element Sum of Squares II
LCP 05. Coin Bonus
LCP 27. 黑盒光线反射
```
- - -
# Grokking Dynamic Programming Patterns for Coding Interviews
* DP
  * 动态规划是通过拆分问题，定义问题状态和状态之间的关系，使得问题能够以递推（或者说分治）的方式去解决。
  * 适用场景
    1. 找Max/Min的问题
    2. 发现可能性的问题
    3. 输出所有解的个数问题
  * 不适用场景
    1. 列出所有具体方案（起码是指数级别的复杂度，通常是递归，backtracking）
    2. 集合问题
  * 考虑
    1. 状态
    2. 转移方程
    3. 初始化条件
    4. 返回结果
  * Tips
    * 如果当前状态只与前面的相关的话，我们都可以通过滚动数组，变量来简化空间复杂度–这种尤其适合不太复杂的动态规划问题，简单的二维DP
    * 当前局部最大+当前值，和当前值的对比，而从决定是继续加还是从新来过
    * 最后的结果有可能是任意一个位置，所以不是简单的return dp[-1]而是max(dp)
    * 简单的累加求和做为DP，则转移方程为res(x,y) = dp[y] - dp[x-1]
    * 熟悉Bit运算和概念，要能发现countbit(n) = countbit(n/2) + n%2这么一个方程，就是说一个数乘2意味着bit位左移一位
    * 用三个dp存2，3，5出现作为乘子的个数
    * 需要额外数组来记录已经出现最长的次数，也就是说如果前面有多个长度相等的连续子串的话，cnt要一直+1
* [What is Dynamic Programming? - Grokking Dynamic Programming Patterns for Coding Interviews](https://www.educative.io/courses/grokking-dynamic-programming-patterns-for-coding-interviews/m2G1pAq0OO0)
* [DP总结 | Joshua](http://joshuablog.herokuapp.com/DP%E6%80%BB%E7%BB%93.html)
* Dynamic Programming (DP) is an algorithmic technique for solving an optimization problem by **breaking it down into simpler subproblems** and utilizing the fact that the optimal solution to the overall problem depends upon the optimal solution to its subproblems.
  * Let’s take the example of the **Fibonacci numbers**. As we all know, Fibonacci numbers are a series of numbers in which each number is the sum of the two preceding numbers. The first few Fibonacci numbers are 0, 1, 1, 2, 3, 5, and 8, and they continue on from there.
* **Characteristics of Dynamic Programming**
  1. Overlapping Subproblems
     * Subproblems are smaller versions of the original problem.
     * Any problem has overlapping sub-problems if finding its solution involves solving the same subproblem multiple times.
  2. Optimal Substructure Property
     * Any problem has optimal substructure property if its overall optimal solution can be constructed from the optimal solutions of its subproblems. 
---
## Dynamic Programming Methods
1. Top-down with Memoization
   * In this approach, we try to solve the bigger problem by recursively finding the solution to smaller sub-problems. 
   * Whenever we solve a sub-problem, we cache its result so that we don’t end up solving it repeatedly if it’s called multiple times. 
   * Instead, we can just return the saved result. This technique of storing the results of already solved subproblems is called **Memoization**.
2. Bottom-up with Tabulation
   * Tabulation is the opposite of the top-down approach and avoids recursion. 
     * In this approach, we solve the problem “bottom-up” (i.e. by solving all the related sub-problems first). 
     * This is typically done by filling up an n-dimensional table. Based on the results in the table, the solution to the top/original problem is then computed.
   * Tabulation is the opposite of Memoization, as in Memoization we solve the problem and maintain a map of already solved sub-problems. 
     * In other words, in memoization, we do it top-down in the sense that we solve the top problem first (which typically recurses down to solve the sub-problems).
```python
3129. Find All Possible Stable Binary Arrays I
```
- - -
P1 单调栈
* Stack
* 以上堆栈形式叫单调栈(monotone stack)，栈内元素单调递增或递减，用其可以实现O(n)时间复杂度求解问题。
  * [力扣](https://leetcode-cn.com/problems/next-greater-element-i/solution/dan-diao-zhan-jie-jue-next-greater-number-yi-lei-w/)
* 见名知意,就是栈中元素,按递增顺序或者递减顺序排列的时候. 单调栈的最大好处就是时间复杂度是线性的,每个元素遍历一次!
  * **单调递增栈可以找到左起第一个比当前数字小的元素:**
    * [LeetCode 下一个最大数系列(503,739,1030) - 知乎](https://zhuanlan.zhihu.com/p/60971978)
  * **单调递减栈可以找到左起第一个比当前数字大的元素:**
* [力扣](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/solution/84-by-ikaruga/)
* 单调栈
  * 单调栈分为单调递增栈和单调递减栈
    * 单调递增栈即栈内元素保持单调递增的栈
    * 同理单调递减栈即栈内元素保持单调递减的栈
  * 操作规则（下面都以单调递增栈为例）
    * 如果新的元素比栈顶元素大，就入栈
    * 如果新的元素较小，那就一直把栈内元素弹出来，直到栈顶比新元素小
  * 加入这样一个规则之后，会有什么效果
    * 栈内的元素是递增的
    * 当元素出栈时，说明这个新元素是出栈元素向后找第一个比其小的元素
      * 举个例子，配合下图，现在索引在 6 ，栈里是 1 5 6 。
      * 接下来新元素是 2 ，那么 6 需要出栈。
      * 当 6 出栈时，右边 2 代表是 6 右边第一个比 6 小的元素。
    * 当元素出栈后，说明新栈顶元素是出栈元素向前找第一个比其小的元素
      * 当 6 出栈时，5 成为新的栈顶，那么 5 就是 6 左边第一个比 6 小的元素。

```cpp
stack<int> st;
for(int i = 0; i < nums.size(); i++)
{
	while(!st.empty() && st.top() > nums[i])
	{
		st.pop();
	}
	st.push(nums[i]);
}
```
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/7e876ae756613053b3432cebc9274e9dbdaafd2e6b8492d37fc34ee98f7655ea-%E5%9B%BE%E7%89%87.png)
* 思路
  1. 对于一个高度，如果能得到向左和向右的边界
  2. 那么就能对每个高度求一次面积
  3. 遍历所有高度，即可得出最大面积
  4. 使用单调栈，在出栈操作时得到前后边界并计算面积

```cpp
    //503. Next Greater Element II
    vector<int> nextGreaterElements(vector<int>& nums) {
        vector<int> res(nums.size(),-1);
        stack<int> st;
        for(int i=nums.size()-1;i>=0;i--) st.push(i);
        for(int i=nums.size()-1;i>=0;i--){
            while(!st.empty()&&nums[i]>=nums[st.top()]) st.pop();
            if(!st.empty()) res[i]=nums[st.top()];
            st.push(i);
        }
        return res;    }

```
* [刷题笔记6（浅谈单调栈） - 知乎](https://zhuanlan.zhihu.com/p/26465701)
* [LeetCode 单调栈 - 知乎](https://zhuanlan.zhihu.com/p/61423849)
* 6/22

```python
907.
739.
503.
1030.
84. Largest Rectangle in Histogram
85. Maximal Rectangle
---1063. Number of Valid Subarrays
42.
496. Next Greater Element I
503. Next Greater Element II

1118. Number of Days in a Month

2944. Minimum Number of Coins for Fruits
```
[卡塔兰数 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E5%8D%A1%E5%A1%94%E5%85%B0%E6%95%B0)
---
https://leetcode.cn/circle/discuss/RvFUtj/
- **灵茶山艾府·高质量题解精选**
  - [codeforces-go/leetcode/SOLUTIONS.md at master · EndlessCheng/codeforces-go](https://github.com/EndlessCheng/codeforces-go/blob/master/leetcode/SOLUTIONS.md)
### 1. 滑动窗口与双指针（定长/不定长/至多/至少/恰好/单序列/双序列/三指针）
- https://leetcode.cn/circle/discuss/0viNMK/
#### 1.1 定长滑动窗口
- [1.1.定长滑动窗口 - 力扣（LeetCode）全球极客挚爱的技术成长平台](https://leetcode.cn/problem-list/Epq25H1W/)
#### 1.2 不定长滑动窗口
- **求最长/最大**
  - [2779. 数组的最大美丽值](https://leetcode.cn/problems/maximum-beauty-of-an-array-after-applying-operation/description/) -
  - [1658. 将 x 减到 0 的最小操作数](https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/description/) -
  - [2831. 找出最长等值子数组](https://leetcode.cn/problems/find-the-longest-equal-subarray/description/) -
  - [2271. 毯子覆盖的最多白色砖块数](https://leetcode.cn/problems/maximum-white-tiles-covered-by-a-carpet/description/) -
  - [2106. 摘水果](https://leetcode.cn/problems/maximum-fruits-harvested-after-at-most-k-steps/description/) -
  - [2555. 两个线段获得的最多奖品](https://leetcode.cn/problems/maximize-win-from-two-segments/description/) -
  - [2009. 使数组连续的最少操作数](https://leetcode.cn/problems/minimum-number-of-operations-to-make-array-continuous/description/) -
- **求最短/最小**
  - [1234. 替换子串得到平衡字符串](https://leetcode.cn/problems/replace-the-substring-for-balanced-string/description/) -
  - [2875. 无限数组的最短子数组](https://leetcode.cn/problems/minimum-size-subarray-in-infinite-array/description/) -
  - [1574. 删除最短的子数组使剩余数组有序](https://leetcode.cn/problems/shortest-subarray-to-be-removed-to-make-array-sorted/description/) -
  - [632. 最小区间](https://leetcode.cn/problems/smallest-range-covering-elements-from-k-lists/description/) -
- **求子数组个数**
  - 
#### 1.3 单序列双指针
#### 1.4 双序列双指针
#### 1.5 三指针
#### 1.6 分组循环

---
### 2. 动态规划（背包/状态机/划分/区间/状压/数位/树形/数据结构优化）
- https://leetcode.cn/circle/discuss/tXLS3i/
- [codeforces-go/leetcode/README.md at master · EndlessCheng/codeforces-go](https://github.com/EndlessCheng/codeforces-go/blob/master/leetcode/README.md)
#### 1. 动态规划入门：从记忆化搜索到递推
- [动态规划入门：从记忆化搜索到递推_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Xj411K7oF/?vd_source=9c4cffb10e23fffa8fe6d124050c8a48)
- Memory search —> HashMap —> @cache
- 课后作业：
  1. [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/) 
  2. [746. 使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/description/) - 
  3. [377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/description/) -
  4. [2466. 统计构造好字符串的方案数](https://leetcode.cn/problems/count-ways-to-build-good-strings/description/) -
  5. [2266. 统计打字方案数](https://leetcode.cn/problems/count-number-of-texts/) -
  6. https://leetcode.cn/problems/delete-and-earn/description/
  7. [198. 打家劫舍](https://leetcode.cn/problems/house-robber/description/) - 
  8. https://leetcode.cn/problems/house-robber-ii/description/
  9. https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/description/
#### 2.【0-1 背包】和【完全背包】，包括【空间优化】以及【至多/恰好/至少】等常见变形题目
- [0-1背包 完全背包_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV16Y411v7Y6?vd_source=9c4cffb10e23fffa8fe6d124050c8a48&spm_id_from=333.788.videopod.sections)
- 選或不選
  - 0-1 背包
    - 目標和 回溯 記憶化搜索 遞推 空間優化：2個數組 1個數組
    - 每個物品**至多**選1個: 考慮i-th物品選或不選
      - `dfs(i,c) = max(dfs(i-1, c), dfs(i-1, c-w[i])+v[i])`
    - 记忆化数组  —>  2维数组 `@cache`
    - 常见变形
      - 至多装capacity 求方案数，最大价值和
      - 恰好装capacity 求方案数，最大，最小价值和
      - 至少装capacity 求方案数，最小价值和
        - `dfs(i, c) = dfs(i-1, c) + dfs(i-1, c-w[i])`
      - 空间优化
        - 改成递推 `f[i][c] = f[i-1][c]+f[i-1][c-w[i]] `
        - 初始化记忆数组+考虑DP的初始状态+递归改成循环
        - `f[(i+1)%2][c] = f[i%2][c]+f[i%2][c-w[i]]` —> O(1)
        - 倒过来计算
          - `for c in range(target. x-1. -1): f[c] = fc[c]+f[x-c]`
  - 完全背包
    - 零錢兌換 同上
    - N种物品，可以重复选择
      - `dfs(i,c) = min(dfs(i-1, c), dfs(i, c-w[i])+1)`
    - 1维数组的空间优化：正序计算即可
- 课后作业：
  1. 和为目标值的最长子序列的长度 https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target/
  2. 分割等和子集 https://leetcode.cn/problems/partition-equal-subset-sum/
  3. 将一个数字表示成幂的和的方案数 https://leetcode.cn/problems/ways-to-express-an-integer-as-sum-of-powers/
  4. 零钱兑换 II https://leetcode.cn/problems/coin-change-ii/
  5. 完全平方数 https://leetcode.cn/problems/perfect-squares/
#### 3. 最长公共子序列 编辑距离
- [最长公共子序列 编辑距离_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1TM4y1o7ug?vd_source=9c4cffb10e23fffa8fe6d124050c8a48&spm_id_from=333.788.videopod.sections)

#### 4. [常用数据结构（前缀和/差分/栈/队列/堆/字典树/并查集/树状数组/线段树）](https://leetcode.cn/circle/discuss/mOr1u6/)
- https://leetcode.cn/circle/discuss/mOr1u6/
---
## Segment tree
- [【数据结构】线段树（Segment Tree）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1cb411t7AM/?spm_id_from=333.337.search-card.all.click&vd_source=9c4cffb10e23fffa8fe6d124050c8a48)
- 【题目】给定一个数组arr，数组可能非常大。在程序运行过程中，你可能要做好几次query和update操作：
  - query(arr, L, R) 表示计算数组arr中，从下标L到下标R之间的所有数字的和。
  - update(arr, i, val) 表示要把arr[i]中的数字改成val。
- 怎样尽可能快地完成一系列query和update的操作？
  - 线段树可以在花费一些额外空间的情况下，把这两个操作的时间复杂度都控制在O(log(n))。
### Why
- query O(N) —> sum_array O(1) —> Segment tree O(logN)
- update O(1) —> sum_array O(N) —> Segment tree O(logN)
### Create a segment tree
- root —> sum
- left —> sum left
- right —> sum right
 ![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2025-02-16%20%E4%B8%8B%E5%8D%882.23.00.png)<!-- {"width":473} -->
### How to use O(logN)
- query [3-5] = [2] + [3-5] = 5+27 = 32
- update [4] —> updated 4 to update to root —> O(logN)
### How to save it to a array
- empty node to make the binary tree to be full
- Heap sort
  - node (5)
  - node.left = 2*node+1 (11)
  - node.right = 2*node+2 (12)
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2025-02-16%20%E4%B8%8B%E5%8D%882.27.14.png)<!-- {"width":617} -->
### Create from the raw array
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2025-02-16%20%E4%B8%8B%E5%8D%882.30.31.png)<!-- {"width":617} -->
- Update (idx, val)
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2025-02-16%20%E4%B8%8B%E5%8D%883.02.32.png)<!-- {"width":617} -->
- Query (L, R)
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2025-02-16%20%E4%B8%8B%E5%8D%883.55.47.png)<!-- {"width":617} -->
```cpp
#include <algorithm>
#include <iostream>
using namespace std;
using ll = long long;

int A[100009], B[100009];
int dp[100009];

#define MAX_LEN 1000

void build_tree(int arr[], int tree[], int node, int start, int end) {
  if (start == end) {
    tree[node] = arr[start];
  } else {
    int mid = (start + end) / 2;
    int left_node = 2 * node + 1;
    int right_node = 2 * node + 2;

    build_tree(arr, tree, left_node, start, mid);
    build_tree(arr, tree, right_node, mid + 1, end);

    tree[node] = tree[left_node] + tree[right_node];
  }
}

void update_tree(int arr[], int tree[], int node, int start, int end, int idx,
                 int val) {
  if (start == end) {
    arr[idx] = val;
    tree[node] = val;
    return;
  }
  int mid = (start + end) / 2;
  int left_node = node * 2 + 1;
  int right_node = node * 2 + 2;
  if (idx >= start && idx <= mid) {
    update_tree(arr, tree, left_node, start, mid, idx, val);
  } else {
    update_tree(arr, tree, right_node, mid + 1, end, idx, val);
  }
  tree[node] = tree[left_node] + tree[right_node];
}

int query_tree(int arr[], int tree[], int node, int start, int end, int L,
               int R) {
  if (end < L || start > R) {
    return 0;
  } else if (L <= start && end <= R) {
    return tree[node];
  } else if (start == end) {
    return tree[node];
  }

  int mid = (start + end) / 2;
  int left_node = 2 * node + 1;
  int right_node = 2 * node + 2;

  int sum_left = query_tree(arr, tree, left_node, start, mid, L, R);
  int sum_right = query_tree(arr, tree, right_node, mid + 1, end, L, R);

  return sum_left + sum_right;
}

int main() {
  int arr[] = {1, 3, 5, 7, 9, 11};
  int size = 6;
  int tree[MAX_LEN] = {0};

  build_tree(arr, tree, 0, 0, size - 1);
  int i;
  for (int i = 0; i < 15; i++) {
    cout << "tree" << i << "=" << tree[i] << endl;
  }

  cout << "update_tree" << endl;
  update_tree(arr, tree, 0, 0, size - 1, 4, 6);
  for (int i = 0; i < 15; i++) {
    cout << "tree" << i << "=" << tree[i] << endl;
  }

  cout << "query_tree" << endl;
  int s = query_tree(arr, tree, 0, 0, size - 1, 2, 5);

  cout << s << endl;

  return 0;
}
```

---