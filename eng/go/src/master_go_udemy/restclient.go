package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
)

func main() {
	response, err := http.Get("https://quotes.rest/qod")

	if err != nil {
		fmt.Println(err)
		return
	}
	defer response.Body.Close()
	contents, err := ioutil.ReadAll(response.Body)

	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(contents))

	fmt.Println("\nALL DONE!")
}
