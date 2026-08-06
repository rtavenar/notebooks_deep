pip install jupyter nbconvert torch torchvision scikit-learn matplotlib transformers datasets diffusers

mkdir -p executed
status=0
for f in tp_seance*_corr.ipynb; do
    echo "=== $f ==="
    if jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=3600 \
        --output-dir=executed \
        "$f"; then
        echo "OK: $f"
    else
        echo "ECHEC: $f"
        status=1
    fi
done
exit $status