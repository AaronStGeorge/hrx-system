#!/usr/bin/env bash

set -euo pipefail

header=""
source=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --header=*)
      header="${1#--header=}"
      shift
      ;;
    --source)
      source="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

printf "generated header\n" > "${header}"
printf "generated source\n" > "${source}"
