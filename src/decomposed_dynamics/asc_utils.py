




def print_npy_contents(path, max_items=10):
        print(f"Inspecting: {path}")
        arr = np.load(path, allow_pickle=True)
        print(f" .npy array: shape={getattr(arr,'shape', None)} dtype={getattr(getattr(arr,'dtype', None),'name', type(arr))}")
        try:
            if hasattr(arr, "size") and arr.size <= max_items:
                print("  contents:", arr)
            else:
                if np.issubdtype(getattr(arr, "dtype", object), np.number):
                    print(f"  min={arr.min()} max={arr.max()} mean={arr.mean()}")
                else:
                    sample = list(arr.flat)[:max_items]
                    print("  sample:", sample)
        except Exception as e:
            print("  could not summarize contents:", e)