#  Grokking the Coding: Patterns for Coding Questions
## Lists
- [[Leetcode question list]]

一面 team 自我介绍 项目经历 做法 发散 模型 原理 公式 用户 新模型 项目提升 基础概念 模型 	
	leetcode 30min 套路题 反转链表 200-300 优化 O(N) DP 
		- 高德 场景 商圈 k-means 公式 常用的 K值 SGD Adam DT control BUG *If else*
		- Deep model Recall vector searching *SQL* Control 
		- 传统模型 推荐大方向
		- 大模型 调度方法 分布式系统 Batch train 
		- 核心部门 大厂简单 脉脉 职场 
二面 team leader 
三面 部长 系统设计 管理知识
- [GitHub - formulahendry/955.WLB: 955 不加班的公司名单 - 工作 955，work–life balance (工作与生活的平衡)](https://github.com/formulahendry/955.WLB)

## Refers
[刷题进阶Tips-分享给那些有刷题经验或工作经验的人|一亩三分地刷题版](https://www.1point3acres.com/bbs/thread-289223-1-1.html)
[コーディング面接対策のために解きたいLeetCode 60問 | 新井康平](https://1kohei1.com/leetcode/)
[Introduction](https://yangshun.github.io/tech-interview-handbook/algorithms/algorithms-introduction/)
 [https://github.com/liyin2015/Algorithms-and-Coding-Interviews](https://github.com/liyin2015/Algorithms-and-Coding-Interviews) 
 [https://1kohei1.com/google/](https://1kohei1.com/google/) 
http://kaiyuzheng.me/dump/notes/interview.pdf
[GitHub - VincentUCLA/LCPython](https://github.com/VincentUCLA/LCPython)
[Introduction · Leetcode](https://pobenliu.gitbooks.io/leetcode/)
[LeetCode/jian-zhi-offer at master · sassyst/LeetCode · GitHub](https://github.com/sassyst/LeetCode/tree/master/jian-zhi-offer)
[GitHub - SeanPrashad/leetcode-patterns: A curated list of 160+ leetcode questions grouped by their common patterns](https://github.com/SeanPrashad/leetcode-patterns)
[GitHub - Dharni0607/Leetcode-Questions: Leetcode question list by companies, includes the premium questions. December 2019 updated](https://github.com/Dharni0607/Leetcode-Questions)
[educative-io-contents/Grokking Dynamic Programming Patterns for Coding Interviews.md at master · asutosh97/educative-io-contents · GitHub](https://github.com/asutosh97/educative-io-contents/blob/master/Grokking%20Dynamic%20Programming%20Patterns%20for%20Coding%20Interviews.md)
	- [Grokking Dynamic Programming Patterns for Coding Interviews - Learn Interactively](https://www.educative.io/courses/grokking-dynamic-programming-patterns-for-coding-interviews?coupon_code=dp-1point3acres&affiliate_id=5749180081373184/)
	- [关于课程”Grokking Dynamic Programming Patterns for Coding Interviews”|一亩三分地公开课版](https://www.1point3acres.com/bbs/thread-503954-1-1.html)
	- [GitHub - ShusenTang/LeetCode: LeetCode solutions with Chinese explanation & Summary of classic algorithms.](https://github.com/ShusenTang/LeetCode)
	- [Introduction - coding practice - advanced topics](https://po-jen-lai.gitbook.io/coding-practice-advanced-topics/)
[Grokking the Coding Interview: Patterns for Coding Questions - Learn Interactively](https://www.educative.io/courses/grokking-the-coding-interview)
- - - -

## Tree basic
	- [力扣](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/solution/python3-die-dai-bian-li-chang-gui-jie-fa-xnim/)
	- Heap https://en.wikipedia.org/wiki/Binary_heap
		- Insert: Average O(1)
		- Extract: log(N)
		- Search: N
		- Delete: logN
	- [递归和迭代的区别 - 知乎](https://zhuanlan.zhihu.com/p/49600594#:~:text=%E9%80%92%E5%BD%92%E6%98%AF%E9%87%8D%E5%A4%8D%E8%B0%83%E7%94%A8%E5%87%BD%E6%95%B0,%E5%BE%AA%E7%8E%AF%E8%AE%A1%E7%AE%97%E7%9A%84%E5%88%9D%E5%A7%8B%E5%80%BC%E3%80%82&text=%E9%80%92%E5%BD%92%E5%BE%AA%E7%8E%AF%E4%B8%AD%EF%BC%8C%E9%81%87%E5%88%B0,%E9%80%90%E5%B1%82%E8%BF%94%E5%9B%9E%E6%9D%A5%E7%BB%93%E6%9D%9F%E3%80%82)
		- 1、程序结构不同
			- 递归是重复调用函数自身实现循环。迭代是函数内某段代码实现循环。 其中，迭代与普通循环的区别是：迭代时，循环代码中参与运算的变量同时是保存结果的变量，当前保存的结果作为下一次循环计算的初始值。
		- 2、算法结束方式不同
			- 递归循环中，遇到满足终止条件的情况时逐层返回来结束。迭代则使用计数器结束循环。 当然很多情况都是多种循环混合采用，这要根据具体需求。
		- 3、效率不同
			- 在循环的次数较大的时候，迭代的效率明显高于递归
- - - -
## 1. Pattern: Sliding window **
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
5/4 5/18 5/27 5/28 5/29 6/12 6/25 7/1 7/25 9/12 9/16 9/21 9/22
```python
*-*-**--*--209 - Minimum Size Subarray Sum
****-***340. Longest Substring with At Most K Distinct Characters
*****-*159.Longest Substring with At Most Two Distinct Characters
******-****904. Fruit Into Baskets
*******3. Longest Substring Without Repeating Characters
*-**-*-*------76. Minimum Window Substring
**-*--*53. Maximum Subarray

***-**------*--424. Longest Repeating Character Replacement
**-**--*--**-------*567. Permutation in String
****-**-*--438. Find All Anagrams in a String
*--*--*----325. Maximum Size Subarray Sum Equals k
--67. Add Binary
--30. Substring with Concatenation of All Words

472. Concatenated Words
727. Minimum Window Subsequence
395.Longest Substring with At Least K Repeating Characters
992. Subarrays with K Different Integers
108. Convert Sorted Array to Binary Search Tree
242. Valid Anagram
```
- - - -
# 2 Pattern: Merge Intervals 
- 7/23 7/25 5/13 5/14 5/15 5/30 6/14 6/27 6/28 9/28
```python
Sort by start
If first.end >= second.start:
	merge

Maybe use heap to store end time
```
```python
********-*56. Merge Intervals
*****-***252. Meeting Rooms

*-*-*----*57. Insert Interval
**-**-**-----986. Interval List Intersections
-*-***----253. Meeting Rooms II
-*---*759. Employee Free Time 
```
- - - -
## 3. Pattern: Two Points *
[Two Pointers - LeetCode](https://leetcode.com/tag/two-pointers/)
	- 4/20 7/19 7/24 5/6 5/29 6/12 6/13 6/14 6/25 6/26 7/1
	- 9/21 9/23
	- Given an array of **sorted** numbers and a target sum, find a pair in the array whose sum is equal to the given target.
	- Consider Hash table to make a memory.
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
```python
# 2 pointer sum question
****-***1. Two Sum
******167. Two Sum II - Input array is sorted
*-***-*-*653. Two Sum IV - Input is a BST
**---*16. 3Sum Closest

--**--*-*-*---*-*15. 3Sum
*-***-----18. 4Sum

# remove resort question *
***27. Remove Element
**-****283. Move Zeroes
*--*-*----*---------75. Sort Colors
**--****-------*26. Remove Duplicates from Sorted Array
**----**----80.Remove Duplicates from Sorted Array II
# others
*********977. Squares of a Sorted Array
****-844. Backspace String Compare

--------*--581.Shortest Unsorted Continuous Subarray
*----**--38. Count and Say
-------713. Subarray Product Less Than K
*--*-*---560. Subarray Sum Equals K
```
- - - -
# 4. Pattern: Fast & Slow pointers *
	- 4/18 7/23 7/24 5/8 5/30 6/12 6/14 6/27 6/28 9/28
	- The Fast & Slow pointer approach, also known as the Hare & Tortoise algorithm, is a pointer algorithm that uses two pointers which move through the array (or sequence/LinkedList) at different speeds. 
	- This approach is quite useful when dealing with cyclic LinkedLists or arrays.
	
```python
slow, fast = head, head
slow = slow.next
fast = fast.next.next
if slow == fast: ---
while fast is not None and fast.next is not None: ---
cyc_len += 1
start cycle: fast = slow + cyc_len; slow == fast
```
```python
*****-*-*141. Linked List Cycle
**-**-***876. Middle of the Linked List

***---***--*---142. Linked List Cycle II
-*-***-**--*202. Happy Number
-*-**-*--*234. Palindrome Linked List
-*--**----143. Reorder List
----------457. Circular Array Loop
-*---*-160. Intersection of Two Linked Lists
# Attention!
-*--*---208. Implement Trie (Prefix Tree)
```
- - - -
# 5 Pattern: Cyclic Sort ***
	- This pattern describes an interesting approach to deal with problems **involving arrays containing numbers in a given range**. 
	- To efficiently solve this problem, we can use the fact that the input array contains numbers in the range of 1 to ‘n’. 
		- For example, to efficiently **sort the array**, we can try placing each number in its correct place, i.e., placing ‘1’ at index ‘0’, placing ‘2’ at index ‘1’, and so on. 
		- Once we are done with the sorting, we can iterate the array to find all indices that are **missing the correct numbers**. 
		- These will be our required numbers.
	- Let’s jump on to our first problem to understand the **Cyclic Sort** pattern in detail.
	- Sort method
		- [Data Structure Visualization](https://www.cs.usfca.edu/~galles/visualization/Algorithms.html)
		- heap rank
			- Time NlogN Space N
		- Quick sort: l + [pivot] + r Add random
			- Time (NlogN) Space (logN)
	- 7/24 5/15 5/31 6/14 6/28 6/29 9/29
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

```python
***-*912. Sort an Array
**-*-*--*--*268. Missing Number
**--*----*287. Find the Duplicate Number
**----*---*442. Find All Duplicates in an Array

-----*----*41. First Missing Positive
--*-*1539. Kth Missing Positive Number

-*-*-*--148. Sort List
*---*-147. Insertion Sort List
----1060. Missing Element in Sorted Array 
```
- - - -
# 6 Pattern: In-place Reversal of a LinkedLis
	- In a lot of problems, we are asked to *reverse* the links between a set of nodes of a LinkedList. 
		- Often, the constraint is that we need to do this in-place, i.e., using the existing node objects and without using extra memory.
		- In-place Reversal of a LinkedList pattern describes an efficient way to solve the above problem. 

	- 5/16 5/17 5/18 6/16 6/18 6/19 6/30 9/29
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
******206. Reverse Linked List
-*---*-*---92. Reverse Linked List II
*-----*-*----25. Reverse Nodes in k-Group
-*---*-*----24. Swap Nodes in Pairs
-*----*-----61. Rotate List
*****83. Remove Duplicates from Sorted List
-----*-82. Remove Duplicates from Sorted List II
--***--19. Remove Nth Node From End of List
***--328. Odd Even Linked List
***-203. Remove Linked List Elements
**--160. Intersection of Two Linked Lists
```
- - - -
# 7 Pattern: Tree Breadth First Search BFS
	- 4/25 5/9 6/1 6/19 6/30
```
Any problem involving the traversal of a tree in a level-by-level order can be efficiently solved using this approach. 

We will use a Queue to keep track of all the nodes of a level before we jump onto the next level. 

This also means that the space complexity of the algorithm will be O(W), where ‘W’ is the maximum number of nodes on any level.
```
	- Time: N
	- Space: N
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

```python
****---*102. Binary Tree Level Order Traversal
***107. Binary Tree Level Order Traversal II
****---103. Binary Tree Zigzag Level Order Traversal
**-*637. Average of Levels in Binary Tree
****515. Find Largest Value in Each Tree Row
**--*-----111. Minimum Depth of Binary Tree
****---104. Maximum Depth of Binary Tree
*----863. All Nodes Distance K in Binary Tree
***-*-116. Populating Next Right Pointers in Each Node
-*117. Populating Next Right Pointers in Each Node II
**-429. N-ary Tree Level Order Traversal
**-199. Binary Tree Right Side View
**-----*-285. Inorder Successor in BST
-*---*--510. Inorder Successor in BST II
```
- - - -
# 8 Pattern: Tree Depth First Search DFS
- DFS approach
- This also means that the space complexity of the algorithm will be 
O(H), where ‘H’ is the maximum height of the tree.
```python

def fun(self, root, target):
		if not root:
			return Fasle
		
		if target == root.val and Blablabla:
			return True 
		
		return self.fun(root.left, target-root.val) or self.fun(root.right, target-root.val)
```
	1. We will keep track of the current path in a list which will be passed to every recursive call.
	2. Whenever we traverse a node we will do two things:
		* Add the current node to the current path.
		* As we added a new node to the current path, we should find the sums of all sub-paths ending at the current node. If the sum of any sub-path is equal to ‘S’ we will increment our path count.
	3. We will traverse all paths and will not stop processing after finding the first path.
	4. Remove the current node from the current path before returning from the function. This is needed to **Backtrack** while we are going up the recursive call stack to process other paths.
	
	- 4/23 4/25 5/10 5/12 5/13 6/2 6/3 6/19 6/20 7/1
```python
**-***-**---112. Path Sum
**--*---113. Path Sum II
**----**------129. Sum Root to Leaf Numbers
*-988. Smallest String Starting From Leaf
****-*257. Binary Tree Paths
*--*---**------437. Path Sum III
*--*-**-----543. Diameter of Binary Tree
----*--687. Longest Univalue Path
*-*-*-----124. Binary Tree Maximum Path Sum
**-*---200. Number of Islands
**---*-1522. Diameter of N-Ary Tree
*--**-366. Find Leaves of Binary Tree
****-872. Leaf-Similar Trees
**589. N-ary Tree Preorder Traversal
**590. N-ary Tree Postorder Traversal
```
- - - -
# 9 Pattern: Two Heaps
	- In many problems, where we are given a set of elements such that we can divide them into two parts. 
		- To solve the problem, we are interested in knowing the smallest element in one part and the biggest element in the other part. 
		- This pattern is an efficient approach to solve such problems.
	- This pattern uses two **Heaps** to solve these problems; 
		- A **Min Heap** to find the smallest element and a **Max Heap** to find the biggest element.
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

	- 4/23 5/18 6/3 6/21 7/2 7/12
```python
***-*--*---295. Find Median from Data Stream
--*--*-*---480. Sliding Window Median  
-*--*-*----502. IPO
----*--436. Find Right Interval
```

[Python：从堆中删除元素 - 程序园](http://www.voidcn.com/article/p-bdykbbrr-bsy.html)
[Python 的 heapq 模块源码分析 | 码农网](https://www.codercto.com/a/49843.html)
- - - -
# 10 Pattern: Subsets BFS ****
	- A huge number of coding interview problems involve dealing with  [Permutations](https://en.wikipedia.org/wiki/Permutation)  and  [Combinations](https://en.wikipedia.org/wiki/Combination)  of a given set of elements. 
		- This pattern describes an efficient **Breadth First Search (BFS)** approach to handle all these problems.
		- Time and space: O(N∗2^N)
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
	- 4/26 5/19 5/20 6/3 6/4 6/5 6/21 6/22 7/2 7/3 7/4 7/12
```python
***-**--*-**---78. Subsets
****--*-*-*-*-90. Subsetzs II
***--**---46. Permutations
***--*-*--784. Letter Case Permutation
***--*-*---22. Generate Parentheses
*----*----*-320. Generalized Abbreviation
**-*--*-*--241. Different Ways to Add Parentheses
---*----96. Unique Binary Search Trees
------*-*--95. Unique Binary Search Trees II
```


## P1 单调栈
	- 以上堆栈形式叫单调栈(monotone stack)，栈内元素单调递增或递减，用其可以实现O(n)时间复杂度求解问题。
	- 见名知意,就是栈中元素,按递增顺序或者递减顺序排列的时候. 单调栈的最大好处就是时间复杂度是线性的,每个元素遍历一次!
		- **单调递增栈可以找到左起第一个比当前数字小的元素:**
			- [LeetCode 下一个最大数系列(503,739,1030) - 知乎](https://zhuanlan.zhihu.com/p/60971978)
		- **单调递减栈可以找到左起第一个比当前数字大的元素:**
	- [力扣](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/solution/84-by-ikaruga/)
	- 单调栈
		- 单调栈分为单调递增栈和单调递减栈
			- 单调递增栈即栈内元素保持单调递增的栈
			- 同理单调递减栈即栈内元素保持单调递减的栈
		- 操作规则（下面都以单调递增栈为例）
			- 如果新的元素比栈顶元素大，就入栈
			- 如果新的元素较小，那就一直把栈内元素弹出来，直到栈顶比新元素小
		- 加入这样一个规则之后，会有什么效果
			- 栈内的元素是递增的
			- 当元素出栈时，说明这个新元素是出栈元素向后找第一个比其小的元素
				- 举个例子，配合下图，现在索引在 6 ，栈里是 1 5 6 。
				- 接下来新元素是 2 ，那么 6 需要出栈。
				- 当 6 出栈时，右边 2 代表是 6 右边第一个比 6 小的元素。
			- 当元素出栈后，说明新栈顶元素是出栈元素向前找第一个比其小的元素
				- 当 6 出栈时，5 成为新的栈顶，那么 5 就是 6 左边第一个比 6 小的元素。
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
		- 思路
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
	- [刷题笔记6（浅谈单调栈） - 知乎](https://zhuanlan.zhihu.com/p/26465701)
	- [LeetCode 单调栈 - 知乎](https://zhuanlan.zhihu.com/p/61423849)
	- 6/22
```python
907.
739.
503.
1030.
84. Largest Rectangle in Histogram
85. Maximal Rectangle
---1063. Number of Valid Subarrays
42.
```
[卡塔兰数 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E5%8D%A1%E5%A1%94%E5%85%B0%E6%95%B0)
- - - -
# 11 Pattern: Modified Binary Search ****
	- As we know, whenever we are given a sorted **Array** or **LinkedList** or **Matrix**, and we are asked to find a certain element, the best algorithm we can use is the  [Binary Search](https://en.wikipedia.org/wiki/Binary_search_algorithm) .
	- This pattern describes an efficient way to handle all problems involving **Binary Search**. 
		- We will go through a set of problems that will help us build an understanding of this pattern so that we can apply this technique to other problems we might come across in the interviews.
	- [二分查找的坑点与总结_haolexiao的专栏-CSDN博客](https://blog.csdn.net/haolexiao/article/details/53541837)
		- 以下这个函数是二分查找nums中[left，right)部分，right值取不到，如果查到的话，返回所在地，如果查不到则返回最后一个小于target值得后一个位置。
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
	- 4/29 5/20 5/21 5/22 6/5 6/22 6/23 7/3 7/4 7/12
```python
*****704. Binary Search
**-*-*-*--744. Find Smallest Letter Greater Than Target
**-*-*-**1150. Check If a Number Is Majority Element in a Sorted Array
***--*-*-*--34. Find First and Last Position of Element in Sorted Array
**-*---*-*---162. Find Peak Element
-**-*----1095. Find in Mountain Array
-*-*--*--*---33. Search in Rotated Sorted Array
**---*-*----81. Search in Rotated Sorted Array II
*-*--*--*---153. Find Minimum in Rotated Sorted Array
-----*---154. Find Minimum in Rotated Sorted Array II

-74. Search a 2D Matrix
540. Single Element in a Sorted Array
702. Search in a Sorted Array of Unknown Size
-1283. Find the Smallest Divisor Given a Threshold
```
- - - -
# 12 Pattern: Bitwise XOR
	- XOR is a logical bitwise operator that returns 0 (false) if both bits are the same and returns 1 (true) otherwise. 
		- In other words, it only returns 1 if exactly one bit is set to 1 out of the two bits in comparison.
![](Grokking%20the%20Coding%20Patterns%20for%20Coding%20Questions/%E6%88%AA%E5%B1%8F2021-05-22%2011.22.21.png)
	- Double numbers
		- Find groups
		- 1 XOR 0 = 0 XOR 1 = 1
		- rightmostSetBit =1 >> 1
	-  4/29 4/30 5/22 6/6 7/3 7/12
``` python
^ same => 0
^ not same => 1
```
```python
******-136. Single Number
-*--137. Single Number II
---**---260. Single Number III
-*-*---1009. Complement of Base 10 Integer
***---832. Flipping an Image
```
- - - -
# 13 Pattern: Top ‘K’ Elements
	- Any problem that asks us to find the top/smallest/frequent ‘K’ elements among a given set falls under this pattern.
		- The best data structure that comes to mind to keep track of ‘K’ elements is **[Heap](https://en.wikipedia.org/wiki/Heap_(data_structure))**. 
		- This pattern will make use of the Heap to solve multiple problems dealing with ‘K’ elements at a time from a set of given elements.
	-  4/30 5/22 5/23 5/24 6/6 6/7 7/5 7/10

``` python
The best data structure that comes to mind to keep track of top ‘K’ elements is Heap. 

Considered Hashtable combo with heap

Time: NLogK
Space: K
```
```python
--****-**215. Kth Largest Element in an Array
***--973. K Closest Points to Origin
***--1167. Minimum Cost to Connect Sticks
***--347. Top K Frequent Elements
***451. Sort Characters By Frequency
***703. Kth Largest Element in a Stream
*--*-*--*--658. Find K Closest Elements
***1133. Largest Unique Number
**-*-*-*--767. Reorganize String
--*---358. Rearrange String k Distance Apart
-*--621. Task Scheduler
*-*--895. Maximum Frequency Stack
--*--692. Top K Frequent Words
```
- - - -
# 14 Pattern: K-way merge
	- This pattern helps us solve problems that involve a list of sorted arrays.
		- Whenever we are given ‘K’ sorted arrays, we can use a **Heap** to efficiently perform a sorted traversal of all the elements of all arrays. 
		- We can push the smallest (first) element of each sorted array in a **Min Heap** to get the overall minimum. 
		- While inserting elements to the **Min Heap** we keep track of which array the element came from. 
		- We can, then, remove the top element from the heap to get the smallest element and push the next element from the same array, to which this smallest element belonged, to the heap. 
		- We can repeat this process to make a sorted traversal of all elements.

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
	- 5/4 5/23 5/24 6/7 7/6
```python
*-***-*-23. Merge k Sorted Lists
******21. Merge Two Sorted Lists
**-*--*-378. Kth Smallest Element in a Sorted Matrix
---**---632. Smallest Range Covering Elements from K Lists
---**---373. Find K Pairs with Smallest Sums

4. Median of Two Sorted Arrays
```
- - - -
# 15 Pattern: Topological Sort (Graph) *
	-  [Topological Sort](https://en.wikipedia.org/wiki/Topological_sorting)  is used to find a linear ordering of elements that have dependencies on each other. 
		- For example, if event ‘B’ is dependent on event ‘A’, ‘A’ comes before ‘B’ in topological ordering.
		- This pattern defines an easy way to understand the technique for performing topological sorting of a set of elements and then solves a few problems using it.
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
	- Time: O(V+E)
	- Space: O(V+E)
5/4 5/5 6/9 7/7 9/30 10/1
```python
****--207.Course Schedule
**--*-*-210.Course Schedule II

----*--269.Alien Dictionary
-*-*--444.Sequence Reconstruction
---310.Minimum Height Trees
```
- - - -
# 16 Pattern: Fibonacci numbers
	- Fibonacci number pattern
		- `CountWays(n) = CountWays(n-1) + CountWays(n-2) + CountWays(n-3) + … + CountWays(n-k), for n >= k`
		- `dp[end] = Math.min(dp[end], dp[start]+1);`
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
	- 7/14 9/29 9/30 10/1
```python
*-**509. Fibonacci Number
*-**70. Climbing Stairs

-*-*--55. Jump Game
----45. Jump Game II
-----746. Min Cost Climbing Stairs
----983. Minimum Cost For Tickets
*--198. House Robber
*--213. House Robber II
--337. House Robber III
```
- - - -
# 17 Pattern : 0/1 Knapsack (Dynamic Programming)
	- **0/1 Knapsack** pattern is based on the famous problem with the same name which is efficiently solved using **Dynamic Programming (DP)**.
		- In this pattern, we will go through a set of problems to develop an understanding of DP. 
		- We will always start with a brute-force recursive solution to see the overlapping subproblems, i.e., realizing that we are solving the same problems repeatedly.
		- After the recursive solution, we will modify our algorithm to apply advanced techniques of **Memoization** and **Bottom-Up Dynamic Programming** to develop a complete understanding of this pattern.
	- 01 背包问题：
		- 最基本的背包问题就是 01 背包问题：一共有 N 件物品，第 i（i 从 1 开始）件物品的重量为 w[i]，价值为 v[i]。在总重量不超过背包承载上限 W 的情况下，能够装入背包的最大价值是多少？
	- 完全背包问题：
		- 完全背包与 01 背包不同就是每种物品可以有无限多个：一共有 N 种物品，每种物品有无限多个，第 i（i 从 1 开始）种物品的重量为 w[i]，价值为 v[i]。
		- 在总重量不超过背包承载上限 W 的情况下，能够装入背包的最大价值是多少？
		- 可见 01 背包问题与完全背包问题主要区别就是*物品是否可以重复选取*。
	- 背包问题具备的特征：
		- 是否可以根据一个 **target**（直接给出或间接求出），target 可以是数字也可以是字符串，再给定一个数组 arrs，问：能否使用 arrs 中的元素做各种**排列组合**得到 target?
	- 背包问题解法：
		- 01 背包问题：
			- 如果是 01 背包，即数组中的元素**不可重复使用**，外循环遍历 arrs，**内循环遍历 target，且内循环倒序**:
		- 完全背包问题：
			- 1. 如果是完全背包，即数组中的元素可重复使用并且不考虑元素之间顺序，arrs 放在外循环（*保证 arrs 按顺序*），target在内循环。且内循环正序。
			- 2. 如果组合问题需考虑元素之间的顺序，**需将 target 放在外循环，将 arrs 放在内循环，且内循环正序**。
	- 5/4 5/23 5/24 6/8 7/6 7/7 7/9 7/12 7/13 7/23 9/30 10/1
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

```python
--**----*--416. Partition Equal Subset Sum
--**--*-*--39. Combination Sum
*-*---*--494. Target Sum
---*-*--139. Word Break
-*--*-*-279. Perfect Squares
-------*--377. Combination Sum IV
*-*---*-322. Coin Change
**---*--518. Coin Change 2
--221. Maximal Square
--1235. Maximum Profit in Job Scheduling
```
[LintCode77：Longest Common Subsequence_网络_卷卷萌的博客-CSDN博客](https://blog.csdn.net/mengmengdajuanjuan/article/details/85372596)
- - - -
# 18 Pattern: Palindromic Subsequence
	- Given a sequence, find the length of its Longest Palindromic Subsequence (LPS). 
	- In a palindromic subsequence, elements read the same backward and forward.
	- Manacher’s Algorithm
	` Minimum_deletions_to_make_palindrome = Length(st) - LPS(st)`
	- **Basic Solution**
		- A basic brute-force solution could be to try all the subsequences of the given sequence. 
		- We can start processing from the beginning and the end of the sequence. So at any step, we have two options:
			1. If the element at the beginning and the end are the same, we increment our count by two and make a recursive call for the remaining sequence.
			2. We will skip the element either from the beginning or the end to make two recursive calls for the remaining subsequence.
		- If option one applies then it will give us the length of LPS; otherwise, the length of LPS will be the maximum number returned by the two recurse calls from the second option.
		- In each function call, we are either having one recursive call or two recursive calls (we will never have three recursive calls); 
			- hence, the time complexity of the above algorithm is exponential O(2^n), where ‘n’ is the length of the input sequence. 
			- The space complexity is O(n), which is used to store the recursion stack.
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
	- **Bottom-up Dynamic Programming **
		- Since we want to try all the subsequences of the given sequence, we can use a two-dimensional array to store our results.
		- We can start from the beginning of the sequence and keep adding one element at a time. 
		- At every step, we will try all of its subsequences. 
		- So for every startIndex and endIndex in the given string, we will choose one of the following two options:
			1. If the element at the startIndex matches the element at the endIndex, the length of LPS would be two plus the length of LPS till startIndex+1 and endIndex-1.
			2. If the element at the startIndex does not match the element at the endIndex, we will take the maximum LPS created by either skipping element at the startIndex or the endIndex.
		- So our recursive formula would be:
```python
if st[endIndex] == st[startIndex] 
  dp[startIndex][endIndex] = 2 + dp[startIndex + 1][endIndex - 1]
else 
  dp[startIndex][endIndex] = Math.max(dp[startIndex + 1][endIndex], dp[startIndex][endIndex - 1])
```
		- The time and space complexity of the above algorithm is O(n^2), where ‘n’ is the length of the input sequence.
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
	- 7/14 7/16 7/18 7/22 10/2
```python
---*-516. Longest Palindromic Subsequence
---647. Palindromic Substrings
----*1312. Minimum Insertion Steps to Make a String Palindrome
--132. Palindrome Partitioning II
-1653. Minimum Deletions to Make String Balanced
*5. Longest Palindromic Substring
-*-409. Longest Palindrome
-1682. Longest Palindromic Subsequence II
-730. Count Different Palindromic Subsequences
1400. Construct K Palindrome Strings
131. Palindrome Partitioning
```
- - - -
# 19 Pattern: Longest Common Substring
	- Given two strings ‘s1’ and ‘s2’, find the length of the longest substring which is common in both the strings.
	- maximum sum increasing subsequence (MSIS)
		- `if num[I] > num[j] => dp[I] = dp[j] + num[I] if there is no bigger MSIS for ‘I’`
	- Shortest common super-sequence (SCS)
```python
if s1[i] == s2[j] 
  dp[i][j] = 1 + dp[i-1][j-1]
else 
  dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1]) 
```
	- Completely match ‘m’ and ‘n’ (the two interleaving strings) with ‘p
```python
dp[mIndex][nIndex] = false
if m[mIndex] == p[pIndex] 
  dp[mIndex][nIndex] = dp[mIndex-1][nIndex]
if n[nIndex] == p[pIndex] 
 dp[mIndex][nIndex] |= dp[mIndex][nIndex-1]
```

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
	- 7/22 7/24 Í
```python
**--1143. Longest Common Subsequence
**-583. Delete Operation for Two Strings
*---300. Longest Increasing Subsequence
---1062. Longest Repeating SubstringÍ
面试题 16.18. Pattern Matching LCCI
-*-72. Edit Distance
--97. Interleaving String
-1092. Shortest Common Supersequence 
**1800. Maximum Ascending Subarray Sum
-674. Longest Continuous Increasing Subsequence
```
- - - -
# Grokking Dynamic Programming Patterns for Coding Interviews
	- [What is Dynamic Programming? - Grokking Dynamic Programming Patterns for Coding Interviews](https://www.educative.io/courses/grokking-dynamic-programming-patterns-for-coding-interviews/m2G1pAq0OO0)
	- Dynamic Programming (DP) is an algorithmic technique for solving an optimization problem by **breaking it down into simpler subproblems** and utilizing the fact that the optimal solution to the overall problem depends upon the optimal solution to its subproblems.
		- Let’s take the example of the **Fibonacci numbers**. As we all know, Fibonacci numbers are a series of numbers in which each number is the sum of the two preceding numbers. The first few Fibonacci numbers are 0, 1, 1, 2, 3, 5, and 8, and they continue on from there.
	- **Characteristics of Dynamic Programming**
		1. Overlapping Subproblems
			- Subproblems are smaller versions of the original problem.
			- Any problem has overlapping sub-problems if finding its solution involves solving the same subproblem multiple times.
		2. Optimal Substructure Property
			- Any problem has optimal substructure property if its overall optimal solution can be constructed from the optimal solutions of its subproblems. 
## Dynamic Programming Methods
		1. Top-down with Memoization
			- In this approach, we try to solve the bigger problem by recursively finding the solution to smaller sub-problems. 
			- Whenever we solve a sub-problem, we cache its result so that we don’t end up solving it repeatedly if it’s called multiple times. 
			- Instead, we can just return the saved result. This technique of storing the results of already solved subproblems is called **Memoization**.
		2. Bottom-up with Tabulation
			- Tabulation is the opposite of the top-down approach and avoids recursion. 
				- In this approach, we solve the problem “bottom-up” (i.e. by solving all the related sub-problems first). 
				- This is typically done by filling up an n-dimensional table. Based on the results in the table, the solution to the top/original problem is then computed.
			- Tabulation is the opposite of Memoization, as in Memoization we solve the problem and maintain a map of already solved sub-problems. 
				- In other words, in memoization, we do it top-down in the sense that we solve the top problem first (which typically recurses down to solve the sub-problems).
- - - -
#study/interview #syudy/interview/coding #study/coding/leetcode

