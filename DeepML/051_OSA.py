# In this problem, you need to implement a function that calculates the Optimal String Alignment (OSA) distance between two given strings. The OSA distance represents the minimum number of edits required to transform one string into another. The allowed edit operations are:

#     Insert a character
#     Delete a character
#     Substitute a character
#     Transpose two adjacent characters

# Each of these operations costs 1 unit.

# Your task is to find the minimum number of edits needed to convert the first string (s1) into the second string (s2).

# For example, the OSA distance between the strings caper and acer is 2: one deletion (removing "p") and one transposition (swapping "a" and "c").

import numpy as np

def OSA(source: str, target: str) -> int:

    dp = np.zeros(shape=(len(source)+1, len(target)+1), dtype= int)
    action = np.zeros(shape=(len(source)+1, len(target)+1))

    # Empty target -> n deletions
    dp[:, 0] = np.array([n for n in range(len(source)+1)])

    # Empty source -> n additions
    dp[0, :] = np.array([n for n in range(len(target)+1)])
    
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            # If matches, nothing had to change
            if (source[i-1] == target[j-1]):
                dp[i][j] = dp[i-1][j-1]
                action[i][j] = 0
                continue

            substitution_cost = dp[i-1][j-1] + 1
            deletion_cost = dp[i-1][j] + 1
            insertion_cost = dp[i][j-1] + 1

            dp[i][j] = min(substitution_cost, insertion_cost, deletion_cost)

            # If transposition has a point
            if (source[i-1]==target[j-2]) and (source[i-2]==target[j-1]):
                transposition_cost = dp[i-2][j-2] + 1
                dp[i][j] = min(transposition_cost, dp[i][j])

    return(dp[-1][-1])

            

### Testing

source = "butterfly"
target = "dragonfly"
distance = OSA(source, target)
print(distance)


source = "caper"
target = "acer"
distance = OSA(source, target)
print(distance)


source = "telescope"
target = "microscope"
distance = OSA(source, target)
print(distance)


source = "london"
target = "paris"
distance = OSA(source, target)
print(distance)

