from argparse import ArgumentParser


def add_counterfactual_variant_flags(parser: ArgumentParser) -> None:
    parser.add_argument("--test_length", action="store_true")
    parser.add_argument("--test_stochasticity", action="store_true")
    parser.add_argument("--test_variations", action="store_true")
    parser.add_argument("--test_p_omission", action="store_true")
    parser.add_argument("--test_bin_size", action="store_true")
    parser.add_argument("--test_iti_hazard", action="store_true")
    parser.add_argument("--test_iti_min", action="store_true")
    parser.add_argument("--test_nITI_microstates", action="store_true")
    parser.add_argument("--test_listen_accuracy", action="store_true")
    parser.add_argument("--test_reward_listen", action="store_true")
    parser.add_argument("--test_grid_size", action="store_true")
    parser.add_argument("--test_tprob", action="store_true")
    parser.add_argument("--test_reward_scheme", action="store_true")
    parser.add_argument("--test_reward_margin", action="store_true")
    parser.add_argument("--test_p_cry_if_hungry", action="store_true")
    parser.add_argument("--test_p_cry_if_full", action="store_true")


def parse_counterfactual_variant(variant_name: str):
    if variant_name == "base":
        return "base", None
    key, value = variant_name.split("=", 1)
    try:
        return key, float(value)
    except ValueError:
        return key, value


def _append_if_missing(variants, variant_name, overrides):
    names = {name for name, _ in variants}
    if variant_name not in names:
        variants.append((variant_name, overrides))


def _dedupe_preserve_order(variants):
    seen = set()
    out = []
    for variant_name, overrides in variants:
        if variant_name in seen:
            continue
        seen.add(variant_name)
        out.append((variant_name, overrides))
    return out


def pick_counterfactual_variants(train_args, args):
    env_name = train_args.environment
    variants = []

    if env_name == "tmaze":
        if args.test_length:
            for length in [20, 30, 40, 50, 60]:
                variants.append((f"tmaze_length={length}", {"length": length}))
            _append_if_missing(
                variants,
                f"tmaze_length={int(train_args.length)}",
                {"length": int(train_args.length)},
            )
            return _dedupe_preserve_order(variants)

        if args.test_stochasticity:
            for stochasticity in [0.1, 0.2, 0.3, 0.4, 0.5]:
                variants.append((f"tmaze_stochasticity={stochasticity}", {"stochasticity": stochasticity}))
            _append_if_missing(
                variants,
                f"tmaze_stochasticity={float(train_args.stochasticity)}",
                {"stochasticity": float(train_args.stochasticity)},
            )
            return _dedupe_preserve_order(variants)

    if env_name == "hike":
        if args.test_variations:
            for variations in [1, 2, 4, 8]:
                variants.append((f"hike_variations={variations}", {"variations": variations}))
            _append_if_missing(
                variants,
                f"hike_variations={train_args.variations}",
                {"variations": train_args.variations},
            )
            return _dedupe_preserve_order(variants)

    if env_name == "starkweather":
        if args.test_p_omission:
            for p_omission in [0.0, 0.1, 0.2, 0.3, 0.4]:
                variants.append((f"starkweather_p_omission={p_omission}", {"p_omission": p_omission}))
            _append_if_missing(
                variants,
                f"starkweather_p_omission={float(train_args.p_omission)}",
                {"p_omission": float(train_args.p_omission)},
            )
            return _dedupe_preserve_order(variants)

        if args.test_bin_size:
            for bin_size in [train_args.bin_size, max(1, train_args.bin_size // 2), train_args.bin_size * 2]:
                variants.append((f"starkweather_bin_size={bin_size}", {"bin_size": bin_size}))
            return _dedupe_preserve_order(variants)

        if args.test_iti_hazard:
            for iti_hazard in [0.01, 0.05, 0.1, 0.2]:
                variants.append((f"starkweather_iti_hazard={iti_hazard}", {"iti_hazard": iti_hazard}))
            _append_if_missing(
                variants,
                f"starkweather_iti_hazard={float(train_args.iti_hazard)}",
                {"iti_hazard": float(train_args.iti_hazard)},
            )
            return _dedupe_preserve_order(variants)

        if args.test_iti_min:
            for iti_min in [0, 5, 10, 20]:
                variants.append((f"starkweather_iti_min={iti_min}", {"iti_min": iti_min}))
            _append_if_missing(
                variants,
                f"starkweather_iti_min={int(train_args.iti_min)}",
                {"iti_min": int(train_args.iti_min)},
            )
            return _dedupe_preserve_order(variants)

        if args.test_nITI_microstates:
            for count in [1, 2, 4, 8]:
                variants.append((f"starkweather_nITI_microstates={count}", {"nITI_microstates": count}))
            _append_if_missing(
                variants,
                f"starkweather_nITI_microstates={int(train_args.nITI_microstates)}",
                {"nITI_microstates": int(train_args.nITI_microstates)},
            )
            return _dedupe_preserve_order(variants)

    if env_name == "tiger":
        if args.test_listen_accuracy:
            for accuracy in [0.55, 0.65, 0.75, 0.85, 0.95]:
                variants.append((f"listen_accuracy={accuracy}", {"listen_accuracy": accuracy}))
            _append_if_missing(
                variants,
                f"listen_accuracy={float(train_args.listen_accuracy)}",
                {"listen_accuracy": float(train_args.listen_accuracy)},
            )
            return _dedupe_preserve_order(variants)

        if args.test_reward_listen:
            for reward_listen in [-3, -1, 1]:
                variants.append((f"reward_listen={reward_listen}", {"reward_listen": reward_listen}))
            _append_if_missing(
                variants,
                f"reward_listen={float(train_args.reward_listen)}",
                {"reward_listen": float(train_args.reward_listen)},
            )
            return _dedupe_preserve_order(variants)

    if env_name == "gridworld":
        if args.test_grid_size:
            for size in [6, 8, 10, 12, 14]:
                variants.append((f"grid_size={size}", {"size": size}))
            _append_if_missing(
                variants,
                f"grid_size={int(train_args.size)}",
                {"size": int(train_args.size)},
            )
            return _dedupe_preserve_order(variants)

        if args.test_tprob:
            for tprob in [0.1, 0.3, 0.5, 0.7, 0.9]:
                variants.append((f"tprob={tprob}", {"tprob": tprob}))
            _append_if_missing(
                variants,
                f"tprob={float(train_args.tprob)}",
                {"tprob": float(train_args.tprob)},
            )
            return _dedupe_preserve_order(variants)

        if args.test_reward_scheme:
            for reward_scheme in ["symmetric", "center", "scaled"]:
                variants.append((f"reward_scheme={reward_scheme}", {"reward_scheme": reward_scheme}))
            _append_if_missing(
                variants,
                f"reward_scheme={train_args.reward_scheme}",
                {"reward_scheme": train_args.reward_scheme},
            )
            return _dedupe_preserve_order(variants)

        if args.test_reward_margin:
            for reward_margin in [0, 2, 4]:
                variants.append((f"reward_margin={reward_margin}", {"reward_margin": reward_margin}))
            _append_if_missing(
                variants,
                f"reward_margin={int(train_args.reward_margin)}",
                {"reward_margin": int(train_args.reward_margin)},
            )
            return _dedupe_preserve_order(variants)

    if env_name == "crybaby":
        if args.test_p_cry_if_hungry:
            for p_cry_if_hungry in [0.30, 0.45, 0.60, 0.75, 0.9]:
                variants.append((f"crybaby_p_cry_if_hungry={p_cry_if_hungry}", {"p_cry_if_hungry": p_cry_if_hungry}))
            _append_if_missing(
                variants,
                f"crybaby_p_cry_if_hungry={float(train_args.p_cry_if_hungry)}",
                {"p_cry_if_hungry": float(train_args.p_cry_if_hungry)},
            )
            return _dedupe_preserve_order(variants)

        if args.test_p_cry_if_full:
            for p_cry_if_full in [0.0, 0.1, 0.2, 0.3, 0.4]:
                variants.append((f"crybaby_p_cry_if_full={p_cry_if_full}", {"p_cry_if_full": p_cry_if_full}))
            _append_if_missing(
                variants,
                f"crybaby_p_cry_if_full={float(train_args.p_cry_if_full)}",
                {"p_cry_if_full": float(train_args.p_cry_if_full)},
            )
            return _dedupe_preserve_order(variants)

    return []
