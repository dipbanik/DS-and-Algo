
# https://leetcode.com/problems/two-sum/ 

# 1. Two Sum
# Easy

# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

# You may assume that each input would have exactly one solution, and you may not use the same element twice.

# You can return the answer in any order.


# Example 1:

# Input: nums = [2, 7, 11, 15], target = 9
# Output: [0, 1]
# Output: Because nums[0] + nums[1] == 9, we return [0, 1].
# Example 2:

# Input: nums = [3, 2, 4], target = 6
# Output: [1, 2]
# Example 3:

# Input: nums = [3, 3], target = 6
# Output: [0, 1]


# Constraints:

# 2 <= nums.length <= 103
# -109 <= nums[i] <= 109
# -109 <= target <= 109
# Only one valid answer exists.

#------------------------------------------------------
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         for val in nums:
#             compliment = target - val
#             if compliment in nums:
#                 ind1 = nums.index(compliment)
#                 ind2 = nums.index(val)
#                 if ind1 != ind2 :
#                     return [ind1,ind2]#[nums.index(compliment), nums.index(val)]
        
#         compliment = nums[ind1]
#         nums[ind1] = None
#         ind2 = nums.index(compliment)
#         return [ind1,ind2]

#---------------------------------------------------
#Solutiom using dictionary mapping 

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        
        for index,value in enumerate(nums):
            if(target-value in d):
                return(d[target-value],index)
            else:
                d[value] = index