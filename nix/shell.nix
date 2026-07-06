{...}: {
  perSystem = {
    config,
    pkgs,
    ...
  }: let
    python = pkgs.python312;
    uv = pkgs.uv;

    commonPackages = [
      uv
      python
      pkgs.cacert
      pkgs.git
      pkgs.just
      pkgs.nil
      pkgs.quarto
    ];

    uvSync = ''
      if [ -f pyproject.toml ]; then
        echo "Synchronizing Schenberg dependencies with uv..."
        uv sync --all-groups
        export PATH="$PWD/.venv/bin:$PATH"
      fi

      ${config.pre-commit.installationScript}
    '';
  in {
    devShells.default = pkgs.mkShell {
      inputsFrom = [config.treefmt.build.devShell];
      name = "schenberg";
      packages = commonPackages;
      shellHook = uvSync;

      env = {
        UV_PYTHON_PREFERENCE = "only-managed";
        PYTHONDONTWRITEBYTECODE = "1";
        SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
      };
    };
  };
}
