from collections import Counter

class Solution:
    def minCost(self, basket1: list[int], basket2: list[int]) -> int:
        
        min_cost = 0

        # Check if the total number of items is odd. Odd number of items cannot be evenly distributed.        
        if ((len(basket1) + len(basket2)) % 2):
            print("Total number of items is odd")
            return -1
        
        # What is the distribution of items in each basket?
        frequency_map_1 = Counter(basket1)
        frequency_map_2 = Counter(basket2)

        # What is the total distribution of items?
        frequency_map = frequency_map_1 + frequency_map_2

                
        # Track the items that need to be moved to balance the baskets.
        moved_set = list()

        # Iterating through all the possible items
        for fruit, count_total in frequency_map.items():
            
            # Cannot distribute items evenly if the total count is odd.
            if count_total % 2:
                return -1
            
            # How many of this (each) item should each basket have at the end?
            count_final = count_total // 2

            # How many of this (each) item needs to be moved from basket 1 to basket 2? Negative if moving from basket 2 to basket 1.
            count_to_move = frequency_map_1[fruit] - count_final
            
            # Add that many of this item to the list.
            for _ in range(0, abs(count_to_move)):
                moved_set.append(fruit)
            
        # For every item that is moved, I can always pick the cheapest item and swap it with the most expensive item. So let's sort the items.
        moved_set.sort()
        print(moved_set)
        
        if (len(moved_set) == 0): 
            return 0
        
        # Note that we use the min value in the frequency map, not in the moved set.
        # This is because the moved set may not contain the cheapest item (C) for later
        min_val = min(frequency_map.keys())
        
        # Since we are moving items in pairs, we only need to consider half of the moved items.
        for i in range(0, len(moved_set) // 2):
            
            # But sometimes the item M that can be nees to be moved is more than twice as expensive than the cheapest item C in the basket.
            # In that case , if we swap N (M's would-be partner in the swap) with C, and then swap M with C.
            # We swapped 2 times at the cost of C (given as the cheapest item) and M was defined to be at least twice as expensive as C.
            # These two swaps are always cheaper than swapping M with N.

            # Note: We don't worry which basket (C) is from because:
            #   - N is larger than M by the logic that it is from the latter half of the moved_set
            #   - M and N can be interchangeable in the swap process.
            
            # So while costing, if using the cheapest item twice is cheaper than moving the original item, we'll use cheapest item instead.
            # Note that this is possible because we don't want the least number of swaps, we want the least cost of swaps.


            min_cost += min(moved_set[i], 2 * min_val)

        return min_cost


sol = Solution()

testCases = [
                [[4,2,2,2], [1,4,1,2], 1],
                [[2,3,4,1], [3,2,5,1], -1],
                [[4,4,4,4,3], [5,5,5,5,3], 8],
                [[84,80,43,8,80,88,43,14,100,88], [32,32,42,68,68,100,42,84,14,8], 48]
            ]

for i, test in enumerate(testCases):
    
    ans = sol.minCost(test[0], test[1])
    if ans == test[2]:
        print(i+1, ": Success")
    else:
        print(i+1, ": Failed - ", ans)

    print("\n\n")