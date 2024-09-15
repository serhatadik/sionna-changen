from sionna_changen.process import SimulationConfig, SimulationRunner

if __name__ == "__main__":
    config = {
        "freq": 1.8e9,
        "synthetic_array": True,
        "tx_array_pattern": "tr38901",
        "tx_array_pol": "V",
        "rx_array_pattern": "dipole",
        "rx_array_pol": "V",
        "tx_loc": [-1178, -310, 91.0],
        "max_depth": 10,
        "edge_diffraction": True,
        "num_samples": 1e7,
        "batch_size_cir": 50,
        "target_num_cirs": 100000,
        "max_gain_db": 0,
        "min_gain_db": -140,
        "min_dist": 10,
        "max_dist": 1300,
        "delay_bins": 250,
        "subcarrier_spacing": 15e3,
        "geo_data_dir": "/home/gtpropagation/Documents/stadik/geo_data/",
        "save_dir": "../cir_data/"
    }

    simulation_config = SimulationConfig(config)
    simulation_config.save_config()

    runner = SimulationRunner(config)
    runner.run_simulation()