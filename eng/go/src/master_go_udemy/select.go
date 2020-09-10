package main

import (
	"fmt"
	"math/rand"
	"time"
)

// An example data table of security clearance
var scMapping = map[string]int{
	"James": 5,
	"Kevin": 10,
	"Rahul": 9,
}

func findsc(name, server string, c chan int) {

	// Simulate a searching action
	time.Sleep(time.Duration(rand.Intn(50)) * time.Minute)

	// Return security clearance from someone
	c <- scMapping[name]
}

//
func main() {

	rand.Seed(time.Now().UnixNano())

	c1 := make(chan int)
	c2 := make(chan int)
	
	name := "James"

	go findsc(name, "Server 1", c1)
	go findsc(name, "Server 2", c2)

	fmt.Println("\nALL DONE!")
}
