def test_table_ts():    

    from telos import Table_TSR
    import cv2

    model = Table_TSR()
    img = cv2.imread("./tests/test_img/test_lore.jpg")
    result = model(img)
    with open(f"output-table-tsr.html", "w", encoding="utf-8") as f:
        f.write(result)
