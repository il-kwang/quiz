import numpy as np

array1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#array1 = 2차원 배열 출력 #3x3

print(array1)
arr1 = array1[:,:1] #1열
arr2 = array1[:,1:2]#2열
arr3 = array1[:1,:] #1행
arr4 = array1[1:2,:] #2행
'''print('-'*30)
print(arr1)
print(arr2)
print(arr3)
print(arr4)
print('-'*30)'''
# array1 첫 번째 column 벡터와 두 번째 column 벡터를 더하여 봅시다. 
array2 = arr1 + arr2
#array1.sum(axis=0)
print("1st column of array1 + 2nd column of array1:\n", array2)

# array1 첫 번째 row 벡터와 두 번째 row 벡터를 빼봅시다.
array3 = arr3 - arr4
print("\n1st row of array1 - 2nd row of array1:\n", array3)

# array2과 array3을 곱하여 봅시다.
array4 = array2 * array3
print("\narray2 * array3:\n", array4)

print(array2.shape)
print(array3.shape)
print(array4.shape)

# array2, array3, array4값을 column으로 이어 붙인 배열
array3_padded = np.tile(array3, (3,1))
array5 = np.vstack((array2.T, array3_padded, array4))
print(array5)

# array1을 array5로 나누어 봅시다.
array1_tile = np.tile(array1, (7//3+1, 1))[:7, :]
array6 = array1_tile / array5
print("\narray1 / array5:\n", array6)
