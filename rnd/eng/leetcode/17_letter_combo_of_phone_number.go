package main

import (
	"fmt"
)

var digitsToLetters = map[string][]string{
	"2": {"a", "b", "c"},
	"3": {"d", "e", "f"},
	"4": {"g", "h", "i"},
	"5": {"j", "k", "l"},
	"6": {"m", "n", "o"},
	"7": {"p", "q", "r", "s"},
	"8": {"t", "u", "v"},
	"9": {"w", "x", "y", "z"},
}

func letterCombinations(digits string) []string {
	if digits == "" {
		return []string{}
	}

	var output []string

	var backtrack func(combination string, nextDigits string)

	backtrack = func(combination string, nextDigits string) {
		if len(nextDigits) >= 1 {
			for _, letter := range digitidTodigitsToLetters[string(nextDigits[0])] {
				backtrack(combination+letter, nextDigits[1:])
			}
		} else {
			output = append(output, combination)
		}
	}

	backtrack("", output)
	return output
}

func main() {
	inputs := map[string][]string{
		"23": {"ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"},
		"":   {},
	}

	for digits, _ := range inputs {
		if len(digits) > 0 && digits[0] == '1' {
			fmt.Printf("digits: %s containt '1'\n", digits)
			continue
		}
		if len(digits) > 4 {
			continue
		}

		out := letterCombinations(digits)
		fmt.Printf("digits: %s, out: %v\n", digits, out)
	}
}
