package session3;

public class InterestChecker {

	// EXERCISE 1
	
	private static final double baseInterest = 0.1;
	private static final double fidelityBonus = 0.1;
	private static final int minNbYearsForFidelityBonus = 3;
	private static final double passwordBonus = 0.2;
	private static final String superSecretPassword = "MOREMONEYPLEASE";
	
	public static double calculateInterest(double accountBalance, int nbYearsCustomer, String bonusCode){
	double totalInterest = baseInterest;
	
	if(nbYearsCustomer >= minNbYearsForFidelityBonus){
	totalInterest += fidelityBonus;
	}
	
	if(bonusCode.equals(superSecretPassword)){
	totalInterest += passwordBonus;
	}
	
	return accountBalance*totalInterest;
	}
}
