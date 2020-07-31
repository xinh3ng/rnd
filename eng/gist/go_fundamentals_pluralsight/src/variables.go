package main

import (
	"fmt"
)

var (
	name, course string
	module       float32 = 3
)

func changeCourse(course string) string {
	course = "A new course of Docker Deep Dive"
	return course
}

// Change the course name by pass-by-reference
func changeCourseByPtr(coursePtr *string) {

	*coursePtr = "A new course of Docker Deep Dive"
}

func main() {

	// a := 10.0  // What is declared inside a function must be used
	b := 3
	course := "Docker Deep Dive"

	fmt.Println("name is", name)
	fmt.Println("module is", module)

	// fmt.Println(b * module) // This is not fine
	fmt.Println(float32(b) * module) // This is not fine
	fmt.Println("\nMemory location of the module variable is", &module)

	fmt.Println("\ncourse before changeCourse() is", course)
	changeCourse(course)
	fmt.Println("course after changeCourse() is", course)

	fmt.Println("\ncourse before changeCourseByPtr() is", course)
	changeCourseByPtr(&course)
	fmt.Println("course after changeCourseByPtr() is", course)

	println("\nALL DONE\n")
}
