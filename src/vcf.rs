use rust_htslib::{bam, bam::Read};


let bam = bam::Reader::from_path(&"test/test.bam").unwrap();
let header = bam::Header::from_template(bam.header());

// print header records to the terminal, akin to samtool
for (key, records) in header.to_hashmap() {
    for record in records {
         println!("@{}\tSN:{}\tLN:{}", key, record["SN"], record["LN"]);
    }
}



// fn read_vcf_genotypes(
// 	chrom: str,
// 	pos: int,
// 	start: int,
// 	end: int,
// 	samples: Vec<str>,
// ) -> Array3<bool> {
// 	...
// }