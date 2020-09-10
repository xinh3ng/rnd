package main

import (
	"fmt"
)

type SLLNode struct {
	value int
	next *SLLNode
}

func (sNode *SLLNode) SetValue(v int) {
	sNode.value = v
}

func (sNode *SLLNode) GetValue() int {
	return sNode.value
}

func NewSLLNode() *SLLNode {
	return new(SLLNode)
}

//
func main() {

	node := NewSLLNode()
	node.SetValue(4)
	fmt.Println("node's value:", node.GetValue())

	fmt.Println("\nALL DONE!\n")
}
