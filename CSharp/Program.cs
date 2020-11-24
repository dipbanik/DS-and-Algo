using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;



namespace Practice
{

    /// <summary>
    /// Breadth First Search and Depth First Search implementation
    /// </summary>
    class Graph
    {
        private int _V;
        LinkedList<int>[] _adj;

        public Graph(int V)
        {
            _adj = new LinkedList<int>[V];
            for (int i = 0; i < _adj.Length; i++)
            {
                _adj[i] = new LinkedList<int>();
            }
            _V = V;
        }

        public void AddEdge(int v, int w)
        {
            _adj[v].AddLast(w);

        }

        public void BFS(int s)
        {
            bool[] visited = new bool[_V];
            for (int i = 0; i < _V; i++)
            {
                visited[i] = false;
            }

            Queue<int> queue = new Queue<int>();

            visited[s] = true;

            queue.Enqueue(s);

            while(queue.Any())
            {
                s = queue.First();
                Console.WriteLine(s + " ");
                queue.Dequeue();

                LinkedList<int> list = _adj[s];

                foreach (var item in list)
                {
                    if(!visited[item])
                    {
                        visited[item] = true;
                        queue.Enqueue(item);
                    }
                }
            }


        }

        public void DFS(int v)
        {
            // Mark all the vertices as not visited 
            // (set as false by default in c#)
            bool[] visited = new bool[_V];

            // Call the recursive helper function to print DFS traversal 
            DFSutil(v, visited);

        }

        private void DFSutil(int v, bool[] visited)
        {
            //mark the current node as visited
            visited[v] = true;
            Console.WriteLine(v + " ");

            //Recur for all the vertices adjacent to the vertex
            LinkedList<int> list = _adj[v];
            foreach (var item in list)
            {
                if (!visited[item])
                    DFSutil(item, visited);
            }
        }

    }


    class Program
    {
        static void Main(string[] args)
        {
            // https://leetcode.com/discuss/interview-question/344650/Amazon-Online-Assessment-Questions
            //Program.TwoSumTrigger();

            //Program.BuySellStocksTrigger();

            //Program.ContainsDuplicateTrigger();

            //Program.ProductExceptSelfTrigger();

            //Program.MaxSubArrayTrigger();

            //Program.MaxProductTrigger();

            //Program.FindMinTrigger();

            //Program.BinarySearchTrigger();

            //Program.SearchRotatedSortedArrayTrigger();

            //Program.ThreeSumTrigger();

            //Program.MaxAreaTrigger();

            //Program.GetSumTrigger();

            //Program.CountingBits();

            //Program.HammingWeightTrigger();

            //Program.MissingNumberTrigger();

            //Program.reverseBitsTrigger();

            //Program.KnapsackTrigger();

            //Program.ClimbingStairsTrigger();

            /*
            Graph g = new Graph(4);

            g.AddEdge(0, 1);
            g.AddEdge(0, 2);
            g.AddEdge(1, 2);
            g.AddEdge(2, 0);
            g.AddEdge(2, 3);
            g.AddEdge(3, 3);

            Console.Write("Following is Breadth First " +
                          "Traversal(starting from " +
                          "vertex 2)\n");
            g.BFS(2);

            Console.ReadLine();

            */

            /*
            Graph g = new Graph(4);

            g.AddEdge(0, 1);
            g.AddEdge(0, 2);
            g.AddEdge(1, 2);
            g.AddEdge(2, 0);
            g.AddEdge(2, 3);
            g.AddEdge(3, 3);

            Console.WriteLine("Following is Depth First Traversal " +
                              "(starting from vertex 2)");

            g.DFS(2);
            Console.ReadKey();
            */


            List<string> str = new List<string>();
            str.Contains("");
            str.Add("item");

            
        }

        #region Climbing Stairs problem
        /*
         You are climbing a stair case. It takes n steps to reach to the top.

            Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

            Example 1:

            Input: 2
            Output: 2
            Explanation: There are two ways to climb to the top.
            1. 1 step + 1 step
            2. 2 steps
            Example 2:

            Input: 3
            Output: 3
            Explanation: There are three ways to climb to the top.
            1. 1 step + 1 step + 1 step
            2. 1 step + 2 steps
            3. 2 steps + 1 step
 

            Constraints:

            1 <= n <= 45
             */

        /// <summary>
        /// You are climbing a stair case. It takes n steps to reach to the top.
        ///Each time you can either climb 1 or 2 steps.In how many distinct ways can you climb to the top?
        /// </summary>
        private static void ClimbingStairsTrigger()
        {
            Program.ClimbingStairs(6);
        }

        private static void ClimbingStairs(int n)
        {
            Console.WriteLine("Brute force method answer is " + Program.climb_Stairs_BruteForce(0,n));
            int[] memo = new int[n+1];
            Console.WriteLine("Recursion with Memoization method answer is " + Program.climb_Stairs_Memo(0, n, memo));
            Console.WriteLine("Dynamic Programmings method answer is " + Program.climbStairsDP(n));
            Console.WriteLine("Fibonacci method answer is " + Program.climbStairsFibonacci(n));

            Console.ReadLine();
        }

        /// <summary>
        /// Recursion with Memoization method
        /// Time complexity : O(n). Size of recursion tree can go upto n.
        ///Space complexity : O(n). The depth of recursion tree can go upto n.
        /// </summary>
        /// <param name="i"></param>
        /// <param name="n"></param>
        /// <param name="memo"></param>
        /// <returns></returns>
        private static int climb_Stairs_Memo(int i, int n, int[] memo)
        {
            if (i > n)
            {
                return 0;
            }
            if (i == n)
            {
                return 1;
            }
            if (memo[i] > 0)
            {
                return memo[i];
            }
            memo[i] = climb_Stairs_Memo(i + 1, n, memo) + climb_Stairs_Memo(i + 2, n, memo);
            return memo[i];
        }

        /// <summary>
        /// Brute Force method for climbing stairs problem.
        /// Time complexity : O(2^n)
        /// Space complexity : O(n)
        /// </summary>
        /// <param name="i"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static int climb_Stairs_BruteForce(int i, int n)
        {
            if (i > n)
            {
                return 0;
            }
            if (i == n)
            {
                return 1;
            }
            return climb_Stairs_BruteForce(i + 1, n) + climb_Stairs_BruteForce(i + 2, n);
        }


        /*
         * dynamic Programming
         * As we can see this problem can be broken into subproblems, and it contains the optimal substructure property i.e. its optimal solution can be constructed efficiently from optimal solutions of its subproblems, we can use dynamic programming to solve this problem.

            One can reach i th
              step in one of the two ways:

            Taking a single step from (i-1)th step.

            Taking a step of 2 from (i-2)th step.

            So, the total number of ways to reach ith is equal to sum of ways of reaching (i-1)th step and ways of reaching 
            (i-2)th step.

            Let dp[i] denotes the number of ways to reach on ith step:

            dp[i]=dp[i-1]+dp[i-2]

            Complexity Analysis

            Time complexity : O(n). Single loop upto n.

            Space complexity : O(n). dp array of size n is used.
         */
        /// <summary>
        /// Dynamic Programming
        /// Time complexity : O(n). Single loop upto n.
        /// Space complexity : O(n). dp array of size n is used.
        /// </summary>
        /// <param name="n">Number of stairs</param>
        /// <returns></returns>
        public static int climbStairsDP(int n)
        {
            if (n == 1)
            {
                return 1;
            }
            int[] dp = new int[n + 1];
            dp[1] = 1;
            dp[2] = 2;
            for (int i = 3; i <= n; i++)
            {
                dp[i] = dp[i - 1] + dp[i - 2];
            }
            return dp[n];
        }

        /*
         * In the above approach we have used dpdp array where dp[i]=dp[i-1]+dp[i-2]. It can be easily analysed that dp[i] is nothing but ith fibonacci number.

            Fib(n)=Fib(n-1)+Fib(n-2)

            Now we just have to find nth number of the fibonacci series having 11 and 22 their first and second term respectively, i.e. Fib(1)=1 and Fib(2)=2
         */
        /// <summary>
        /// Fibonacci Number
        /// Time complexity : O(n). Single loop upto nn is required to calculate nth fibonacci number.
        /// Space complexity : O(1). Constant space is used.
        /// </summary>
        /// <param name="n">number upto which fibonacci or number of Stairs</param>
        /// <returns></returns>
        public static int climbStairsFibonacci(int n)
        {
            if (n == 1)
            {
                return 1;
            }
            int first = 1;
            int second = 2;
            for (int i = 3; i <= n; i++)
            {
                int third = first + second;
                first = second;
                second = third;
            }
            return second;
        }

        
        #endregion


        #region 0-1 Knapsack Problem | DP-10
        /* Problem Statement :
            Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack. In other words, given two integer arrays val[0..n-1] and wt[0..n-1] which represent values and weights associated with n items respectively. Also given an integer W which represents knapsack capacity, find out the maximum value subset of val[] such that sum of the weights of this subset is smaller than or equal to W. You cannot break an item, either pick the complete item or don’t pick it (0-1 property). 
         */

        private static void KnapsackTrigger()
        {
            int W = 50;
            int[] wt = { 10, 20, 30};
            int[] val = { 60, 100, 120 };
            int n = 3;

            Console.WriteLine(Program.knapSackDP(W,wt,val,n));

            Console.ReadLine();

        }

        // Returns the maximum value that can 
        // be put in a knapsack of capacity W 
        public static int knapSack(int W, int[] wt,
                            int[] val, int n)
        {

            // Base Case 
            if (n == 0 || W == 0)
                return 0;

            // If weight of the nth item is 
            // more than Knapsack capacity W, 
            // then this item cannot be 
            // included in the optimal solution 
            if (wt[n - 1] > W)
                return knapSack(W, wt, val, n - 1);

            // Return the maximum of two cases: 
            // (1) nth item included 
            // (2) not included 
            else
                return Math.Max( val[n - 1] + 
                    knapSack(W - wt[n - 1], wt, val, n - 1),
                    knapSack(W, wt, val, n - 1));
        }

        public static int knapSackDP(int W, int[] wt,
                            int[] val, int n)
        {
            int[,] t = new int[n+1,W+1];
            if (n == 0 || W == 0)
                return 0;
            for(int i =0; i<=n; i++)
            {
                for(int j=0; j<=W;j++)
                {
                    if (i == 0 || j == 0)
                        t[i, j] = 0;
                    else if (wt[i - 1] <= j)
                    {
                        t[i,j] = Math.Max(val[i-1] + t[i-1, j - wt[i-1]], t[i-1, j]);
                    }
                    else if (wt[i - 1] > j)
                        t[i, j] = t[i-1,j];
                }
            }
            return t[n,W];
        }

    #endregion
        
    #region GCD

        public static int generalizedGCD(int num, int[] arr)
        {
            // WRITE YOUR CODE HERE
            
            arr.Min();

            return -1;
        }

        #endregion

        #region Reverse Bits
        /*
        Reverse bits of a given 32 bits unsigned integer.



        Example 1:

        Input: 00000010100101000001111010011100
        Output: 00111001011110000010100101000000
        Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
        Example 2:

        Input: 11111111111111111111111111111101
        Output: 10111111111111111111111111111111
        Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.



        Note:

        Note that in some languages such as Java, there is no unsigned integer type.In this case, both input and output will be given as signed integer type and should not affect your implementation, as the internal binary representation of the integer is the same whether it is signed or unsigned.
        In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above the input represents the signed integer -3 and the output represents the signed integer -1073741825.
 

        Follow up:

        If this function is called many times, how would you optimize it?
            */

        private static void reverseBitsTrigger()
        {
            Console.WriteLine(Program.reverseBits(00000000000000000000000010000000));
            //There is some provlem with this.
            Console.ReadLine();
        }

        public static uint reverseBits(uint n)
        {
            uint ans = 0;
            for (int i = 0; i < 32; i++)
            {
                ans <<= 1;
                ans = ans | ((uint)(n & 1));
                n >>= 1;
            }
            return ans;
        }
        #endregion

        #region Missing Number

        /*

            Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

            Example 1:

            Input: [3,0,1]
            Output: 2
            Example 2:

            Input: [9,6,4,2,3,5,7,0,1]
            Output: 8
            Note:
            Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?

          */



        private static void MissingNumberTrigger()
        {
            int[] nums = { 3, 0, 1 };
            Console.WriteLine(Program.MissingNumber(nums));
            Console.WriteLine(Program.MissingNumberSum(nums));
            Console.WriteLine(Program.MissingNumberBitManipulation(nums));
            Console.ReadLine();
        }


        public static int MissingNumber(int[] nums)
        {
            if (nums is null || nums.Length == 0)
                return -1;
            bool[] dic = new bool[nums.Length];
            Dictionary<int, bool> map = new Dictionary<int, bool>();
            //Dictionary<int, bool> temp_map = new Dictionary<int, bool>(map);

            for (int i = 0; i < nums.Length; i++)
            {
                map.Add(nums[i], true);
            }


            for (int i = 0; i <= nums.Length; i++)
            {
                if (!map.ContainsKey(i))
                    return i;
                
            }
            
            return -1;
        }

        public static int MissingNumberBitManipulation(int[] nums)
        {
            int missing = nums.Length;
            for (int i = 0; i < nums.Length; i++)
            {
                missing ^= i ^ nums[i];
            }
            return missing;
        }

        public static int MissingNumberSum(int[] nums)
        {
            int expectedSum = nums.Length * (nums.Length + 1) / 2;
            int actualSum = nums.Sum();             
            return expectedSum - actualSum;
        }



        #endregion

        #region Hamming Weight or Number of 1 bits

        private static void HammingWeightTrigger()
        {
            Console.WriteLine(Program.HammingWeight(00000000000000000000000010000000));
            Console.ReadLine();
        }                   

        public static int HammingWeight(uint n)
        {
            int ones = 0;
            //var nw = Convert.ToInt32(n);
            while (n != 0)
            {
                ones = ones + (int)(n & 1);
                n = n >> 1; //Something is wrong here. Works online though. 
            }
            return ones;
        }

        #endregion


        #region Counting Bits for each number upto n
        /*
         Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

            Example 1:

            Input: 2
            Output: [0,1,1]
            Example 2:

            Input: 5
            Output: [0,1,1,2,1,2]
            Follow up:

            It is very easy to come up with a solution with run time O(n*sizeof(integer)). But can you do it in linear time O(n) /possibly in a single pass?
            Space complexity should be O(n).
            Can you do it like a boss? Do it without using any builtin function like __builtin_popcount in c++ or in any other language.




         * */

        public static void CountingBits()
        {
            int[] nums = Program.CountBits(2);
            Console.Write("[ ");
            foreach (var item in nums)
            {
                Console.Write(item + ", ");
            }
            Console.Write("]");
            Console.ReadLine();
        }

        public static int[] CountBits(int num)
        {
            int[] f = new int[num + 1];
            for (int i = 1; i <= num; i++)
                f[i] = f[i >> 1] + (i & 1);
            return f;
        }
    


    #endregion

    #region Sum of Two Numbers
    /*
    Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.

    Example 1:

    Input: a = 1, b = 2
    Output: 3
    Example 2:

    Input: a = -2, b = 3
    Output: 1
    */

    public static void GetSumTrigger()
        {
            Console.WriteLine(Program.GetSum(-2, 3) );
            Console.ReadLine();
        }

        public static int GetSum(int a, int b)
        {
            int c;
            while (b != 0)
            {
                c = (a & b);
                a = a ^ b;
                b = (c) << 1;
            }
            return a;
            
        }


        #endregion

        #region Container with most water
        /*
        Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate(i, ai). n vertical lines are drawn such that the two endpoints of line i is at(i, ai) and(i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

        Note: You may not slant the container and n is at least 2.
        
        The above vertical lines are represented by array[1, 8, 6, 2, 5, 4, 8, 3, 7]. In this case, the max area of water (blue section) the container can contain is 49.

        Example:

        Input: [1,8,6,2,5,4,8,3,7]
        7*(8-1)
        7*7
        Output: 49

*/

        public static void MaxAreaTrigger()
        {

            int[] nums = { 1, 2, 1 }; 
            //int[] nums = { 1, 8, 6, 2, 5, 4, 8, 3, 7 };
            Console.WriteLine(Program.MaxArea(nums));
            Console.ReadLine();

        }


        public static int MaxArea(int[] height)
        {
            int maxArea = 0;
            int i = 0, j = height.Length - 1;
            while (i < j)
            {
                maxArea = Math.Max(maxArea, Math.Min(height[i], height[j]) * (j - i));
                if (height[i] < height[j])
                    i++;
                else
                    j--;
            }

            return maxArea;
        }

        #endregion

        #region 3-sum problem
        /*
         * Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

            Note:

            The solution set must not contain duplicate triplets.

            Example:

            Given array nums = [-1, 0, 1, 2, -1, -4],

            A solution set is:
            [
              [-1, 0, 1],
              [-1, -1, 2]
            ]
         */

        public static void ThreeSumTrigger()
        {
            int[] nums = { 1,2,3,4,5,6,7};
            var output = Program.ThreeSum(nums);
            foreach (var item in output)
            {
                Console.WriteLine(item);
            }
            Console.ReadLine();
        }

        public static IList<IList<int>> ThreeSum(int[] nums)
        {
            IList<IList<int>> ls = new List<IList<int>>();
            int l = nums.Length;
            Array.Sort(nums);
            int[] p = new[] { int.MaxValue, int.MaxValue };

            for (int i = 0; i < l - 2; ++i)
            {
                if (p[0] == nums[i])
                    continue;
                p[0] = nums[i];
                p[1] = int.MaxValue;
                int k = l - 1;
                for (int j = i + 1; j < k; ++j)
                {
                    if (p[1] == nums[j])
                        continue;

                    while (j < k && nums[i] + nums[j] + nums[k] < 0)
                        ++j;
                    while (j < k && nums[i] + nums[j] + nums[k] > 0)
                        --k;
                    if (j < k && nums[i] + nums[j] + nums[k] == 0)
                        ls.Add(new List<int> { nums[i], nums[j], nums[k] });
                    p[1] = nums[j];
                }
            }

            return ls;
        }

        #endregion

        #region Search for number in Rotated Sorted Array

        /*
         *  Question : 
         *  
            Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

            (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

            You are given a target value to search. If found in the array return its index, otherwise return -1.

            You may assume no duplicate exists in the array.

            Your algorithm's runtime complexity must be in the order of O(log n).

            Example 1:

            Input: nums = [4,5,6,7,0,1,2], target = 0
            Output: 4
            Example 2:

            Input: nums = [4,5,6,7,0,1,2], target = 3
            Output: -1
         */

        public static void SearchRotatedSortedArrayTrigger()
        {
            int[] input = { 7, 8, 9, 10, 1, 2,3,4,5,6};
            int searchval = 7;

            Console.WriteLine(  Program.SearchRotatedSortedArrayLeet(input, searchval));
            Console.ReadLine();
        }


        /*
         * SOLUTION:
         *  the main idea is that we need to find some parts of array that we could adopt binary search on that, which means we need to find some completed sorted parts, then determine whether target is in left part or right part. There is at least one segment (left part or right part) is monotonically increasing.
            
            If the entire left part is monotonically increasing, which means the pivot point is on the right part
            If left <= target < mid ------> drop the right half
            Else ------> drop the left half
            If the entire right part is monotonically increasing, which means the pivot point is on the left part
            If mid < target <= right ------> drop the left half
            Else ------> drop the right half
         */


        public static int SearchRotatedSortedArrayLeet(int[] nums, int target)
        {
            if (nums == null || nums.Length == 0)
            {
                return -1;
            }

            /*.*/
            int left = 0, right = nums.Length - 1;
            //when we use the condition "left <= right", we do not need to determine if nums[left] == target
            //in outside of loop, because the jumping condition is left > right, we will have the determination
            //condition if(target == nums[mid]) inside of loop
            while (left <= right)
            {
                //left bias
                int mid = left + (right - left) / 2;
                if (target == nums[mid])
                {
                    return mid;
                }
                //if left part is monotonically increasing, or the pivot point is on the right part
                if (nums[left] <= nums[mid])
                {
                    //must use "<=" at here since we need to make sure target is in the left part,
                    //then safely drop the right part
                    if (nums[left] <= target && target < nums[mid])
                    {
                        right = mid - 1;
                    }
                    else
                    {
                        //right bias
                        left = mid + 1;
                    }
                }

                //if right part is monotonically increasing, or the pivot point is on the left part
                else
                {
                    //must use "<=" at here since we need to make sure target is in the right part,
                    //then safely drop the left part
                    if (nums[mid] < target && target <= nums[right])
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }
            }
            return -1;
        }


        #endregion


        #region Binary Search

        public static void BinarySearchTrigger()
        {
            int[] input = { 0, 1, 2, 4, 5, 6, 7 };
            int val = 5;
            Console.WriteLine(Program.BinarySearch(input, val));
            Console.ReadLine();
        }


        public static bool BinarySearch(int[] nums, int val)
        {
            if (nums.Length == 0)
                return false;
            else if (nums.Length == 1)
            {
                if (nums[0] == val)
                    return true;
            }
            else if (nums.Length > 1)
            {
                int left = 0, right = nums.Length - 1;
                while (left <= right)
                {
                    int mid = left + (right - left) / 2;
                    if (nums[mid] == val)
                        return true;
                    if (nums[mid] > val)
                        right = mid - 1;
                    else
                        left = mid + 1;
                }
            }
            
            return false;
        }



        #endregion


        #region Minimum in rotated sorted array
        /*
        Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

        (i.e., [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

        Find the minimum element.

        You may assume no duplicate exists in the array.

        Example 1:

        Input: [3,4,5,1,2]
        Output: 1
        Example 2:

        Input: [4,5,6,7,0,1,2]
        Output: 0
        
        */

        public static void FindMinTrigger()
        {
            int[] input = { };// 4, 5, 6, 7, 0, 1, 2 };

            Console.WriteLine(Program.FindMinBinary(input));
            Console.ReadLine();
        }

        public static int FindMinBinary(int[] nums)
        {
            // If the list has just one element then return that element.
            if (nums.Length == 1)
            {
                return nums[0];
            }

            // initializing left and right pointers.
            int left = 0, right = nums.Length - 1;

            // if the last element is greater than the first element then there is no rotation.
            // e.g. 1 < 2 < 3 < 4 < 5 < 7. Already sorted array.
            // Hence the smallest element is first element. A[0]
            if (nums[right] > nums[0])
            {
                return nums[0];
            }

            // Binary search way
            while (right >= left)
            {
                // Find the mid element
                int mid = left + (right - left) / 2;

                // if the mid element is greater than its next element then mid+1 element is the smallest
                // This point would be the point of change. From higher to lower value.
                if (nums[mid] > nums[mid + 1])
                {
                    return nums[mid + 1];
                }

                // if the mid element is lesser than its previous element then mid element is the smallest
                if (nums[mid - 1] > nums[mid])
                {
                    return nums[mid];
                }

                // if the mid elements value is greater than the 0th element this means
                // the least value is still somewhere to the right as we are still dealing with elements
                // greater than nums[0]
                if (nums[mid] > nums[0])
                {
                    left = mid + 1;
                }
                else
                {
                    // if nums[0] is greater than the mid value then this means the smallest value is somewhere to
                    // the left
                    right = mid - 1;
                }
            }
            return -1;
        }

        public static int FindMin(int[] nums)
        {

            if (nums.Length == 0)
                return 0;
            else if (nums.Length == 1)
                return nums[0];
            else if (nums[0] < nums[nums.Length - 1])
                return nums[0];
            else
            {
                for (int i = 0; i < nums.Length - 1; i++)
                {
                    if (nums[i] > nums[i + 1])
                    {
                        return nums[i + 1];
                    }
                }
                return 0;
            }
        }

        #endregion

        #region Maximum Product Subarray

        /*
         Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

            Example 1:

            Input: [2,3,-2,4]
            Output: 6
            Explanation: [2,3] has the largest product 6.
            Example 2:

            Input: [-2,0,-1]
            Output: 0
            Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
         */

        public static void MaxProductTrigger()
        {
            int[] input = { 2, 3, -2, 1, 2, -3, 1, -2, 4 };
            
            Console.WriteLine(Program.MaxProduct(input));
            Console.ReadLine();
        }

        public static int MaxProduct(int[] nums)
        {
            int max = Int32.MinValue, imax = 1, imin = 1;
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] < 0) { int tmp = imax; imax = imin; imin = tmp; }
                imax = Math.Max(imax * nums[i], nums[i]);
                imin = Math.Min(imin * nums[i], nums[i]);

                max = Math.Max(max, imax);
            }
            return max;
        }

     #endregion



        #region MaxSubArray
            /*
             * Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

                Example:

                Input: [-2,1,-3,4,-1,2,1,-5,4],
                Output: 6
                Explanation: [4,-1,2,1] has the largest sum = 6.
                Follow up:

                If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

                Corner Cases -
                [-1]
                [-35,-4,-5,-1]
             * 
             * */
            //https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/

            private static void MaxSubArrayTrigger()
        {
            int[] input = { -2, 1, -3, 4, -1, 2, 1, -5, 4 };
            Console.WriteLine(  Program.MaxSubArray(input));
            Console.ReadLine();
        }

        public static int MaxSubArray(int[] nums)
        {
            int maxSoFar = nums[0], maxEndingHere = 0;
            foreach (var item in nums)
            {
                maxEndingHere = maxEndingHere + item;
                if (maxEndingHere > maxSoFar)
                    maxSoFar = maxEndingHere;
                if (maxEndingHere < 0)
                {
                    maxEndingHere = 0;
                }
            }
            return maxSoFar;


            ////Array.Sort() Uses QuickSort and hence O(nlogn) time complexity.
            //Array.Sort(nums);
            //int sum = 0;
            ////Array.Reverse(nums);

            //for (int i = nums.Length-1; i >= 0; i--)
            //{
            //    if(nums[i]>=0 )
            //    {
            //        sum = sum + nums[i];
            //    }
            //}
            //if(sum == 0)
            //{
            //    sum = nums.Max();
            //}
            //return sum;
        }

            #endregion

        #region ProductExcept Self

            /*
             * Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

                Example:

                Input:  [1,2,3,4]
                Output: [24,12,8,6]
                Constraint: It's guaranteed that the product of the elements of any prefix or suffix of the array (including the whole array) fits in a 32 bit integer.

                Note: Please solve it without division and in O(n).

                Follow up:
                Could you solve it with constant space complexity? (The output array does not count as extra space for the purpose of space complexity analysis.)
             * */
            private static void ProductExceptSelfTrigger()
        {

            //Corner Cases - Multiple zeros and single zero in the input set.
            int[] nums = { 1, 2, 3, 4 };
    
            var output = Program.ProductExceptSelf(nums);

            foreach (var item in output)
            {
                Console.Write(item + " ");
            }
            Console.ReadLine();
        }

        /*With less space complexity. 
         * 
         */ 
        public static int[] ProductExceptSelf(int[] nums)
        {
            int[] output = new int[nums.Length];
            output[0] = 1;
            for (int i = 1; i < nums.Length; i++)
            {
                output[i] = output[i - 1] * nums[i - 1];
            }
            int R = 1;
            for (int i = nums.Length - 1; i >= 0; i--)
            {

                // For the index 'i', R would contain the 
                // product of all elements to the right. We update R accordingly
                output[i] = output[i] * R;
                R *= nums[i];
            }
            return output;
        }

        /*
         * Left and Right Products List approach.
         * O(n) space and O(n) time complexity
         * */
        //public static int[] ProductExceptSelf(int[] nums)
        //{
        //    int[] Left = new int[nums.Length];
        //    int[] Right = new int[nums.Length];
        //    for (int i = 0, j = nums.Length - 1; i < nums.Length; i++, j--)
        //    {
        //        if (i == 0)
        //        {
        //            Left[i] = 1;
        //            Right[j] = 1;
        //        }
        //        else
        //        {
        //            Left[i] = Left[i - 1] * nums[i - 1];
        //            Right[j] = Right[j + 1] * nums[j + 1];
        //        }
        //    }
        //    for (int i = 0; i < nums.Length; i++)
        //    {
        //        nums[i] = Left[i] * Right[i];
        //    }

        //    return nums;
        //}

        /*Not Applicable as I am using division here.
         * */
        //public static int[] ProductExceptSelf(int[] nums)
        //{
        //    int fullProduct = 1, count = 0;
        //    if (nums is null)
        //    {
        //        return nums;
        //    }
        //    foreach (var item in nums)
        //    {
        //        if (item != 0)
        //            fullProduct *= item;
        //        else
        //            count++;

        //    }

        //    for (int i = 0; i < nums.Length; i++)
        //    {
        //        if (count == 0)
        //            nums[i] = fullProduct / nums[i];
        //        else if (count == 1)
        //        {
        //            if (nums[i] == 0)
        //                nums[i] = fullProduct;
        //            else
        //                nums[i] = 0;
        //        }
        //        else if (count > 1)
        //            nums[i] = 0;
        //    }

        //    return nums;

        //}

        #endregion


        #region ContainsDuplicate

        private static void ContainsDuplicateTrigger()
        {
            /*
             * Given an array of integers, find if the array contains any duplicates.

                Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

                Example 1:

                Input: [1,2,3,1]
                Output: true
                Example 2:

                Input: [1,2,3,4]
                Output: false
                Example 3:

                Input: [1,1,1,3,3,4,3,2,4,2]
                Output: true
             * 
             * */

            int[] nums = { 1, 1, 1, 3, 3, 4, 3, 2, 4, 2 };
            nums = new int[] { 1, 2, 3, 4 };
            Console.WriteLine(Program.ContainsDuplicate(nums));
            Console.ReadLine();


        }

        /*
         * Time COmplexity is O(n)
         * Space Complexity is O(n)
         * For certain test cases with not very large n, the runtime of this method can be slower than Sorting and then linear searching which takes O(nlogn) time and O(1) space. The reason is hash table has some overhead in maintaining its property. One should keep in mind that real world performance can be different from what the Big-O notation says. The Big-O notation only tells us that for sufficiently large input, one will be faster than the other. Therefore, when n is not sufficiently large, an O(n) algorithm can be slower than an O(nlogn) algorithm.
        */
        private static bool ContainsDuplicate(int[] nums)
        {
            //int[] output = new int[nums.Length];
            Dictionary<int, int> map = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                if (map.ContainsKey(nums[i]))
                    return true;
                else
                    map.Add(nums[i], 1);
            }
            return false;
        }

        #endregion


        #region BuySell Stock
        private static void BuySellStocksTrigger()
        {
            /*
             * 
             * Say you have an array for which the ith element is the price of a given stock on day i.

                If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

                Note that you cannot sell a stock before you buy one.

                Example 1:

                Input: [7,1,5,3,6,4]
                Output: 5
                    Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
                    Not 7-1 = 6, as selling price needs to be larger than buying price.
                Example 2:

                Input: [7,6,4,3,1]
                Output: 0
                Explanation: In this case, no transaction is done, i.e. max profit = 0.
             * 
             */

            //int[] prices = { 7, 1, 5, 3, 6, 4 };
            int[] prices = { 7, 6, 4, 3, 1 };
            int profit = Program.MaxProfit(prices);

            Console.WriteLine("The profit is {0}", profit);
            Console.ReadLine();

        }

        
        /*
         * Time Complexity is O(n)
         * Space Complexity is O(1) 
        */
        private static int MaxProfit(int[] prices)
        {
            int profit = 0, minprice = 0;

            for (int i = 0; i < prices.Length; i++)
            {
                if (i == 0)
                    minprice = prices[i];
                else
                {
                    if (prices[i] < minprice)
                    {
                        minprice = prices[i];
                    }
                    else
                    {
                        if (prices[i] - minprice > profit)
                            profit = prices[i] - minprice;
                    }
                }
            }
            return profit;
        }

        #endregion


        #region TwoSumProblem
        private static void TwoSumTrigger()
        {
            int[] arr = new int[2];
            arr = Program.TwoSum(new int[] { 2, 7, 11, 15 }, 17);
            if (arr is null)
            {
                Console.WriteLine("No match found!");
            }
            else
            {
                foreach (var item in arr)
                {
                    Console.WriteLine(item);

                }
            }
            Console.ReadLine();
        }

        /*
        //For Brute Focrce - space complexity - O(1) Time Complexity - O(n^2)
        //For mapper function - Time complexity : O(n). We traverse the list containing nn elements exactly twice. Since the hash table reduces the look up time to O(1), the time complexity is O(n).
           Space complexity : O(n). The extra space required depends on the number of items stored in the hash table, which stores exactly nn elements.
        */
        public static int[] TwoSum(int[] nums, int target)
        {

            //for (int i = 0; i < nums.Length-1; i++)
            //{
            //     for(int j = i + 1; j< nums.Length; j++)
            //    {
            //        if (nums[i] + nums[j] == target)
            //        {
            //            return new int[] { i, j };                        
            //        }
            //    }
            //}
            //return null;

            Dictionary<int, int> map = new Dictionary<int, int>();

            //for (int i = 0; i < nums.Length; i++)
            //    map.Add(nums[i], i);

            for (int i = 0; i < nums.Length; i++)
            {
                int complement = target - nums[i];
                if (map.ContainsKey(complement))// && map[complement] != i)
                {
                    return new int[] { map[complement], i };
                }
                map.Add(nums[i], i);
            }
            map.Max();
            throw new Exception("No two sum solution");
        }


        #endregion


        #region Reverse String
        static string reverseString(string inputString)
        {
            return String.Join(" ", inputString.Split(' ').Select(x => new String(x.Reverse().ToArray())));
        }

        #endregion

    }



    #region Inheritence
    interface Vehicle
    {

    }

    class LandVehicles : Vehicle
    {

    }

    class SeaVehicles : Vehicle
    {

    }

    class Car : LandVehicles
    {

    }

    class Ship : SeaVehicles
    {

    }

    #endregion Inheritence


    #region Test Classes
    class Test<T>
    {
        Test(T val)
        {
            this._value = val;
        }
        T _value;



        void printhere()
        {
            Console.WriteLine(_value);
        }
    }

    #endregion

}

