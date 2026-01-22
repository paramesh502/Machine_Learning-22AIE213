
import numpy as np

def vowels_consonants(word):
    vowels = "aeiouAEIOU"
    v_count = 0
    c_count = 0
    for x in word:
        if x in vowels:
            v_count += 1
        else:
            c_count += 1
    return v_count, c_count

def multiply_matrics(A, B):
    a = np.array(A)
    b = np.array(B)
    if a.shape[1] != b.shape[0]:
        return "Error"
    result = np.dot(a, b)
    return result

def common_elements(l1,l2):
    s1 = set(l1)
    s2 = set(l2)
    common = s1.intersection(s2)
    return len(common)

def get_transpose(matrix):
    m = np.array(matrix)
    transpose = np.transpose(m)
    return transpose

def mean_mode_median(numbers):
    numbers.sort()
    total = sum(numbers)
    mean = total / len(numbers)
    median = (numbers[49] + numbers[50])/2
    freq = {}
    for n in numbers:
        freq[n] = freq.get(n, 0) + 1
    mode = max(freq, key=freq.get)
    return mean, mode, median

if __name__ == "__main__":
    word = "paramesh"
    v,c = vowels_consonants(word)
    print("que1:")
    print(f"Vowels: {v}, Consonants: {c}")   

    A = [[1, 2, 3], [4, 5, 6]]
    B = [[3], [2], [1]]
    print("que2:")
    print("Matrix Multiplication Result:")
    print(multiply_matrics(A, B))   

    l1 = [18,45,7,1]
    l2 = [17,18,333]
    print("que3:")
    print(f"Common elements count: {common_elements(l1, l2)}")

    mtrix = [[1,5,7], [0,0,0], [7,5,3]]
    print("que4:")
    print("Transpose of matrix:")
    print(get_transpose(mtrix))

    numbers = list(np.random.randint(100,150,100))
    mean, mode, median = mean_mode_median(numbers)
    print("que5:")
    print(numbers)
    print(f"Mean: {mean}, Mode: {mode}, Median: {median}")
