use std::fs;
use std::ops::Index;

use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2_field::goldilocks_field::GoldilocksField;

/// An example of using Plonky2 to prove a statement of the form
/// "I know the 100th element of the Fibonacci sequence, starting with constants a and b."
/// When a == 0 and b == 1, this is proving knowledge of the 100th (standard) Fibonacci number.
/// This example also serializes the circuit data and proof to JSON files.
fn main() -> Result<()> {
    // const MAX_POW: u32 = 24;
    // let num_cycles = (0..=MAX_POW).map(|i| 1 << i).collect::<Vec<u64>>();

    // for (i, nc) in num_cycles.into_iter().enumerate() {
    //     println!("i = {i}");
    //     let circuit_data = build_cicruit(nc);
    //     println!(
    //         "degree_bits: {}\n",
    //         circuit_data.common.fri_params.degree_bits
    //     )
    // }

    // return Ok(());

    ///////////////////////////////////////////////////////////////////////////////

    const POW: u32 = 15;
    let num_cycles: u64 = 1 << POW;

    let (data, proof) = build_cicruit(num_cycles);

    let common_circuit_data_serialized = serde_json::to_string(&data.common).unwrap();
    fs::write("common_circuit_data.json", common_circuit_data_serialized)
        .expect("Unable to write file");

    let verifier_only_circuit_data_serialized = serde_json::to_string(&data.verifier_only).unwrap();
    fs::write(
        "verifier_only_circuit_data.json",
        verifier_only_circuit_data_serialized,
    )
    .expect("Unable to write file");

    let proof_serialized = serde_json::to_string(&proof).unwrap();
    fs::write("proof_with_public_inputs.json", proof_serialized).expect("Unable to write file");

    // println!(
    //     "100th Fibonacci number mod |F| (starting with {}, {}) is: {}",
    //     proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
    // );

    data.verify(proof)

    ///////////////////////////////////////////////////////////////////////////////

    // let data = build_cicruit();

    // let common_circuit_data_serialized = serde_json::to_string(&data.common).unwrap();
    // fs::write("common_circuit_data.json", common_circuit_data_serialized)
    //     .expect("Unable to write file");

    // let verifier_only_circuit_data_serialized = serde_json::to_string(&data.verifier_only).unwrap();
    // fs::write(
    //     "verifier_only_circuit_data.json",
    //     verifier_only_circuit_data_serialized,
    // )
    // .expect("Unable to write file");

    // let proof = data.prove(pw)?;

    // let proof_serialized = serde_json::to_string(&proof).unwrap();
    // fs::write("proof_with_public_inputs.json", proof_serialized).expect("Unable to write file");

    // println!(
    //     "100th Fibonacci number mod |F| (starting with {}, {}) is: {}",
    //     proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
    // );

    // data.verify(proof)
}

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

fn build_cicruit(
    num_cycles: u64,
) -> (
    CircuitData<GoldilocksField, PoseidonGoldilocksConfig, D>,
    ProofWithPublicInputs<GoldilocksField, PoseidonGoldilocksConfig, D>,
) {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let mut nums = vec![];

    // The arithmetic circuit.
    let mut sum = builder.add_virtual_target();

    for _ in 0..num_cycles {
        let num = builder.add_virtual_target();
        nums.push(num);
        sum = builder.add(sum, num);
    }

    // Public input is just the sum (which is generated).
    builder.register_public_input(sum);

    // Provide initial values.
    let mut pw = PartialWitness::new();
    pw.set_target(sum, F::ZERO).unwrap();
    for i in 0..nums.len() {
        pw.set_target(nums[i], Field::from_canonical_usize(i + 1))
            .unwrap();
    }

    println!(
        "num_cycles = {},\tnum_gates = {:?}",
        num_cycles,
        builder.num_gates()
    );

    let data = builder.build::<C>();
    let proof = data.prove(pw).unwrap();

    (data, proof)
}
