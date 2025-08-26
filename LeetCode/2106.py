class Solution:
    def maxTotalFruits(self, fruits: list[list[int]], startPos: int, k: int) -> int:

        if not len(fruits):
            return 0
        
        left = max(0, startPos - k)
        right = startPos + (k - (startPos - left))

        max_count = 0
        running_count = 0
        l = 0
        r = 0
        
        # Iterate through the fruits to find the first fruit in the window
        while l < len(fruits) and fruits[l][0] < left:
            l += 1
            r += 1
            
        # Iterate through the window to count fruits in the left-most window
        while r < len(fruits) and fruits[r][0] <= right:
            max_count += fruits[r][1]
            r += 1
        
        running_count = max_count
        while (right < startPos + k):

            if (left < (startPos - (k/2))):
                left += 2
                right += 1
            elif (left < startPos):
                left += 1
                right += 1
            elif left == startPos:
                right += 1
            else:
                left += 1
                
            
            # Move the left pointer to the right
            while (l < r) and (fruits[l][0] < left):
                running_count -= fruits[l][1]
                l += 1

            while (r < len(fruits)) and (r < (startPos + k)) and (fruits[r][0] <= right):
                running_count += fruits[r][1]
                r += 1
            
            max_count = max(max_count, running_count)
            print(left, right, max_count)
            

        return max_count
            
sol = Solution()

testCases = [
                [9, [[[2,8],[6,3],[8,6]], 5, 4]],
                [14, [[[0,9],[4,1],[5,7],[6,2],[7,4],[10,9]], 5, 4]],
                [0, [[[0,3],[6,4],[8,5]], 3, 2]],
                [22, [[[1,9],[2,10],[3,1],[5,6],[6,3],[8,2],[9,2],[11,4],[18,10],[22,8],[25,2],[26,2],[30,4],[31,5],[33,9],[34,1],[39,10]], 19, 9]],
                # [10000, [[[200000, 10000]], 0, 200000]]
            ]

for i, test in enumerate(testCases):
    
    print("- ", (i+1), " -")
    print(test[1][1], test[1][2])
    
    out = []
    j = 0
    for fruit in test[1][0]:
        while j < fruit[0]:
            out.append(0)
            j += 1
        out.append(fruit[1])
        j += 1  
    
    idx = []
    for j in range(len(out)):
        idx.append(j)
    print(idx)
    print(out)
    
    ans = sol.maxTotalFruits(test[1][0], test[1][1], test[1][2])
    if ans == test[0]:
        print("Success")
    else:
        print("Failed - ", ans)

    print("\n\n")