package main

import (
	"fmt"
)

// BinarySearch
func BinarySearch(array []int, target int, method string) int {

	if method == "recursion" {
		highIndex := len(array) - 1
		return binarySearchHelper(array, target, 0, highIndex)

	} else if method == "iteration" {
		return iterBinarySearch(array, target)
	} else {
		panic("Unknown method")
	}
}

// BinarySearchHelper uses recursion
func binarySearchHelper(array []int, target int, lowIndex int, highIndex int) int {

	if lowIndex > highIndex {
		panic(fmt.Sprintf("lowIndex: %d is larger than highIndex: %d, which is not acceptable", lowIndex, highIndex))
	}
	mid := int((highIndex + lowIndex) / 2)

	if array[mid] > target { // target should be in the lower half
		return binarySearchHelper(array, target, lowIndex, mid)
	} else if array[mid] < target { // target should be in the bigger half
		return binarySearchHelper(array, target, mid+1, highIndex)
	} else {
		return mid
	}
}

// IterBinarySearchHelper uses iteration (a while loop)
func iterBinarySearch(array []int, target int) int {

	// startIndex, endIndex := 0, len(array)-1
	// var mid int

	for true {
		break
	}
	return -1
}

//
func main() {

	arr := []int{0, 1, 2, 3, 4, 5}

	var idx int
	for var element int in {1, 2} {
		idx = BinarySearch(arr, element, "recursion")
		fmt.Println(fmt.Sprintf("Using recusion method.  Index of element %d is %d", element, idx))

		idx = BinarySearch(arr, element, "iteration")
		fmt.Println(fmt.Sprintf("Using iteration method. Index of element %d is %d", element, idx))

	}


	fmt.Println("\nALL DONE!")
}
