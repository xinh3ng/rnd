package main

import (
	"fmt"
)

//
func main() {
	c := make(chan bool)
	s := "world"

	go waitAndSay(c, s)
	fmt.Println("hello")

	c <- true   // Give channel c a true signal 
	<-c  // Wait to receive another signal on c

	fmt.Println("\nALL DONE!")
}

func waitAndSay(c chan bool, s string) {
	if b := <-c; b {  // Do something when b receives a true signal
		fmt.Println(s)
	}
	c <- true
}
