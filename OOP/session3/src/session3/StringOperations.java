package session3;

public class StringOperations {
	
	public static String reverse(String word) {
		String reversedWord = "";
		int length = word.length();
		
		// Iterate through word, start at the end, append each character
		for(int i = length - 1; i >= 0; i--) {
			reversedWord += word.charAt(i);
		}
		
		return reversedWord;
	}
	
	public static boolean isPalindrome(String word) {
		
		return reverse(word).equals(word);
	}
	
	public static void main(String[] args) {
		String testWord = "gravitational waves";
		String reversedWord = reverse(testWord);
		
		System.out.println("When you're not drunk: " + testWord);
		System.out.println("When you're drunk: " + reversedWord);
		
		String secretePalindromeQ = "lol";
		System.out.println("The word " + secretePalindromeQ + " is palindrome? " + isPalindrome(secretePalindromeQ));
	}
}
