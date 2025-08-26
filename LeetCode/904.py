class Solution:
    def totalFruit(self, fruits: list[int]) -> int:

        if (len(fruits) == 0):
            return 0
                
        # Let there be 2 fruit_count B1 & B2 that at any point carry N1 number of F1 fruits and N2 number of F2 fruits respectively
        max_count = 0
        fruit_count = [0, 0]
        running_count = [0, 0]
        picked_fruits = [-1, -1]

        # Going through the trees, we pick fruits in either of thw two fruit_count
        for fruit in fruits:

            # Account a fruit if picking
            if (fruit == picked_fruits[0]):
                fruit_count[0] += 1
                running_count[0] += 1
                running_count[1] = 0
                continue

            if (fruit == picked_fruits[1]):
                fruit_count[1] += 1
                running_count[1] += 1
                running_count[0] = 0
                continue

            # A new fruit has been encountered that we weren't picking

            # Update the max_count if necessary
            max_count = max(max_count, (fruit_count[0] + fruit_count[1]))

            # Empty the basket that you are no longer tracking.
            if (running_count[0]):
                picked_fruits[1] = fruit
                fruit_count = [running_count[0], 1]
                running_count = [0, 1]
            else:
                picked_fruits[0] = fruit
                fruit_count = [1, running_count[1]]
                running_count = [1, 0]

        return max(max_count, (fruit_count[0] + fruit_count[1]))

sol = Solution()

testCases = [
                [3, [1,2,1]],
                [3, [0,1,2,2]],
                [4, [1,2,3,2,2]],
                [0, []]
            ]

for i, test in enumerate(testCases):
    
    ans = sol.totalFruit(test[1])
    if ans == test[0]:
        print(i+1, ": Success")
    else:
        print(i+1, ": Failed - ", ans)

    print("\n\n")