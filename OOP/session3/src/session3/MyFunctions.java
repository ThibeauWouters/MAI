package session3;

public class MyFunctions {


	public static int f(double x) {
		int i;
		int total;

		if(x < 0) {
			return -1;
		}

		total = 1;
		i = 1;

		while (i < x) {
			i++;
			total *= i;
		}

		return total;
	}

	public static int[] fList(int y) {
		// y is the length of the list to be returned

		int[] fValues = new int[y];

		for(int j = 0; j < y; j++) {
			fValues[j] = f(j);
		}

		return fValues;
	}
}
