package session3;

public class Patterns {

	public static void upperTriangle(int num) {
		while(num > 0) {
			String line = "";
			for(int i = 0; i < num; i++) {
				line += "*";
			}
			System.out.println(line);
			num--;
		}
	}

	/// /// /// TAKEN FROM SOLUTIONS
	static void lowerTriangle(int num){
		for(int i=1;i<=num;i++){
			for(int j=num;j>i;j--){
				System.out.print(" ");
			}
			for(int j=1;j<=i;j++){
				System.out.print("*");
			}
			System.out.print("\n");
		}
		System.out.print("\n");
	}
	static void diamond(int num){
		for(int i=num-1;i>=0;i--){
			for(int j=0;j<=i;j++)
			{
				System.out.print(" ");
			}
			for(int j=1;j<(num-i)*2;j++)
			{
				System.out.print("*");
			}
			System.out.print("\n");
		}
		for(int i=1;i<num;i++){
			for(int j=0;j<=i;j++)
			{
				System.out.print(" ");
			}
			for(int j=1;j<(num-i)*2;j++)
			{
				System.out.print("*");
			}
			System.out.print("\n");
		}
	}
	
	/// /// /// 
	

	public static void main(String [] args){
		upperTriangle(3);
		lowerTriangle(10);
		diamond(6);
	}
}