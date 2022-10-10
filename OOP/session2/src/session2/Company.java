package session2;

public class Company {

	// Variables
	String companyName;
	String locationName;
	
	
	// Constructor
	Company(String compName, String locName){
		
		companyName = compName;
		locationName = locName;
	}
	
	// Methods
	boolean isBelgian() {
		
		return (locationName.equals("Belgian"));
//		if (locationName.equals("Belgian")){
//			return true;
//		}
//		else {
//			return false;
//		}
	}
	
	public static void main(String[] args) {
		
		Company firstCompany = new Company("Tesla", "Belgian");
		
		System.out.println(firstCompany.isBelgian());
	}
}
