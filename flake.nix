{
  description = "TI shell";

  inputs = {
    nixpkgs.url = "github:NixOs/nixpkgs/nixos-unstable";
  };

  outputs = inputs: let
    system = "x86_64-linux";
    pkgs = import inputs.nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in {
    devShells.${system}.default = let
      python =
        pkgs.python312.withPackages
        (ps:
          with ps; [
            matplotlib
            numpy
            sounddevice
          ]);
    in
      pkgs.mkShell {
        packages = [python];
      };
  };
}
