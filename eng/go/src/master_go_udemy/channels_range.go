package main

import (
	"fmt"
)

//
func main() {

	c := make(chan string)
	n := 5

	go sayHelloMultipleTimes(c, n)
	for s := range c {
		fmt.Println(s)
	}
	v, ok := <-c
	fmt.Println("channel open?", ok, ", value", v)

	fmt.Println("\nALL DONE!")
}

func sayHelloMultipleTimes(c chan string, n int) {
	for i := 0; i < n; i++ {
		c <- "hello"
	}
	close(c) //
}
