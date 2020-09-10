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

	// Return security clearance level of name
	c <- scMapping[name]
}

//
func main() {

	rand.Seed(time.Now().UnixNano())

	c1 := make(chan int)
	c2 := make(chan int)
	name := "James" // The name to be searched

	go findsc(name, "Server 1", c1)
	go findsc(name, "Server 2", c2)

	select {
	case sc := <-c1:
		fmt.Println(name, "has a security clearance level of ", sc, " found from Server 1")
	
	case sc := <-c2:
		fmt.Println(name, "has a security clearance level of ", sc, " found from Server 2")
	 
	case <-time.After(100 * time.Millisecond):
		fmt.Println("Search has timed out")
	}

	fmt.Println("\nALL DONE!")
}
