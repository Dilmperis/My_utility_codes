from itertools import product
import math

# Letter poll to choose from
first = ['a', 'b', 'c']          
second = ['m', 'n', 'o']    
third = ['j', 'i', 'l']          
fourth = ['t', 'u', 'v']    

combinations = [''.join(p) for p in product(first, second, third, fourth)]

# How many Combinations are produced?
len_first, len_second, len_third, len_fourth = len(first), len(second), len(third), len(fourth)

num_combinations = math.prod([len_first, len_second, len_third, len_fourth])
assert num_combinations == len(combinations), 'Not correct number of combinations. Check again the math'
print("Assertion passed. Total combinations:", num_combinations)

'''
Because we have 4 letter words and each letter is coming from
a poll of 3 numbers we have 3^4 =81 combinations
'''

# Display the combinations:
display_comb = input(f'Do you want to see all the {num_combinations} combinations? [y/n]: ')
while display_comb.lower() not in ['y', 'yes', 'n', 'no']:
    display_comb = input(f'Try again one of the two options: [y/n]? ')

if display_comb.lower() in ['y', 'yes']:
    print(combinations)
elif display_comb.lower() in ['n', 'no']:
    print('End of Combinatorial Generation')


