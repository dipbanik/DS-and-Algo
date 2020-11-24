//https://leetcode.com/problems/reverse-integer/

/*
Given a 32-bit signed integer, reverse digits of an integer.

Note:
Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.

Example 1:

Input: x = 123
Output: 321
Example 2:

Input: x = -123
Output: -321
Example 3:

Input: x = 120
Output: 21
Example 4:

Input: x = 0
Output: 0
 

Constraints:

-231 <= x <= 231 - 1

*/

public class Solution {
    public int Reverse(int x) {
        int rev = 0, backup=x, temp=0;
        while(x!=0)
        {
            temp=x%10;
            x=x/10;
            if (rev > Int32.MaxValue/10 || (rev == Int32.MaxValue / 10 && temp > 7)) 
                return 0;
            if (rev < Int32.MinValue/10 || (rev == Int32.MinValue / 10 && temp < -8)) 
                return 0;
            rev = rev*10 + temp;
        }
        return rev;
    }
}