class Solution:
    def maxAdjacentDistance(self, nums: list[int]) -> int:

        retval = 0
        for i in range(0, len(nums)):
            retval = max(retval, abs(nums[ (i+1) % len(nums)] - nums[i]))

        return retval

sol = Solution()

testCases = [
                [3, [1,2,4]],
                [5, [-5,-10,-5]],
                [4, [6,5,5,4,2]],
                [0, []]
            ]

for i, test in enumerate(testCases):
    
    ans = sol.maxAdjacentDistance(test[1])
    if ans == test[0]:
        print(i+1, ": Success")
    else:
        print(i+1, ": Failed - ", ans)

    print("\n\n")