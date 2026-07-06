{inputs, ...}: {
  imports = [inputs.treefmt-nix.flakeModule];
  perSystem = {
    pkgs,
    config,
    ...
  }: {
    formatter = config.treefmt.programs.alejandra.package;
    treefmt = {
      # enabled to be the base formatter
      flakeFormatter = true;

      # pre commit already makes this check
      flakeCheck = true;
      projectRootFile = "flake.nix";

      programs = {
        # Nix
        nixfmt = {
          enable = false;
          package = pkgs.alejandra;
        };

        alejandra = {
          enable = true;
        };

        deadnix = {
          enable = true;
        };

        # Github Actions
        actionlint = {
          enable = true;
        };

        ruff-format = {
          enable = true;
        };

        # Bash
        beautysh = {
          enable = true;
        };

        # TypeScript / JSON
        # biome = { enable = true; };

        # YAML
        prettier = {
          enable = true;
        };
      };
    };
  };
}
