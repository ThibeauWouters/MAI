package alignments;

import java.util.ArrayList;
//import java.util.HashMap;

public class TestStandardAlignment {
	public static void main(String[] args) {
		// Read the fasta file
		FastaContents hiv = new FastaContents("hiv.fasta");

		// Store them in the alignment
		StandardAlignment sa = new StandardAlignment(hiv);
		System.out.println(sa.getIdentifiers());
		
		// Do the tests
		
		System.out.println("---------------------");
		System.out.println("Testing difference score. Before computing:");
		System.out.println(sa.getDifferenceScore(sa.getGenome(">2002.A.CD.02.KTB035")));
		
		/* All tests come here */
		System.out.println("---------------------");
		System.out.println("Testing search genome");
		ArrayList<String> test = sa.searchGenomes("AAAAAAA");
		System.out.println(test);

		System.out.println("---------------------");
		System.out.println("Testing add genome");
		sa.addGenome("newID", "XXX");
		System.out.println(sa.getGenome("newID"));

		System.out.println("---------------------");
		System.out.println("Testing getting genome");
		String test2 = sa.getGenome(">2002.A.CD.02.KTB035");
		System.out.println(test2);

		System.out.println("---------------------");
		System.out.println("Testing getting id");
		String test3 = sa.getIdentifier("XXX");
		System.out.println(test3);

		System.out.println("---------------------");
		System.out.println("Testing replacing genome");
		sa.replaceGenome("newID", "newerID", "ZZZ");
		System.out.println(sa.getGenome("newerID"));

		System.out.println("---------------------");
		System.out.println("Testing replaceSequenceGenome");
		sa.replaceSequenceGenome(">2002.A.CD.02.KTB035", "A", "X");
		System.out.println(sa.getGenome(">2002.A.CD.02.KTB035"));

		System.out.println("---------------------");
		int size = sa.getSize();
		System.out.println("Testing remove genome: before removal " + size);
		sa.removeGenome("heyheyhey");
		size = sa.getSize();
		System.out.println("Removed heyheyhey: " + size);

		sa.removeGenome("newerID");
		size = sa.getSize();
		System.out.println("Removed newerID: " + size);
		test2 = sa.getGenome("newerID");
		System.out.println(test2);

		System.out.println("---------------------");
		System.out.println("Testing is valid sequence");
		boolean valid1 = StandardAlignment.isValidSequence("ACTGATCGATCGATC");
		System.out.println(valid1);
		boolean valid2 = StandardAlignment.isValidSequence("ACXGATCGATCGATC");
		System.out.println(valid2);

		System.out.println("---------------------");
		System.out.println("Testing replaceSequenceGenome. Before:");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(0)));
		sa.replaceSequenceGenome(sa.getIdentifiers().get(0), "TTT", "AAA");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(0)));

		System.out.println("---------------------");
		System.out.println("Testing replaceSequenceAlignment. Before:");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(40)));
		sa.replaceSequenceAlignment("T", "A");
		System.out.println(sa.getGenome(sa.getIdentifiers().get(40)));
		
		System.out.println("---------------------");
		System.out.println("Testing SA/SNiP constructors taking other SA/SNiP alignments.");
		StandardAlignment sa2 = new StandardAlignment(sa);
		System.out.println(sa2.getGenome(">2002.A.CD.02.KTB035"));
		System.out.println(sa2.getIdentifiers());
		
		SNiPAlignment snip = new SNiPAlignment(sa, ">2002.A.CD.02.KTB035");
		System.out.println(snip.getReferenceGenome());
		System.out.println(sa.getGenome(sa.getIdentifiers().get(1)));
		System.out.println(sa2.getGenome(sa2.getIdentifiers().get(1)));
		System.out.println(snip.getGenome(snip.getIdentifiers().get(1)));
		
		StandardAlignment sa3 = new StandardAlignment(snip);
		System.out.println(sa3.getGenome(">2002.A.CD.02.KTB035"));
		
		System.out.println("---------------------");
		System.out.println("Testing the display method");
		sa.display();
	}

}
