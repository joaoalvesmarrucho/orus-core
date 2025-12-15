package main

import "fmt"


func ConvertInterfaceToStrings(input []interface{}) []string {
	result := make([]string, 0, len(input))
	for _, val := range input {
		if str, ok := val.(string); ok {
			result = append(result, str)
		} else {
			fmt.Printf("Warning: The value '%v' is not a string.\n", val)
		}
	}
	return result
}