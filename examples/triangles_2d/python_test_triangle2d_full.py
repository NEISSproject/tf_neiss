#!python
import generate_train_data_2d_triangle_TFR

if __name__ == "__main__":
    #     --to_log_file True --mode "val" --data_id triangle2d_32_dummy
    # Todo: add other call from test_triangle2d_full.sh to replace entirely.
    arg_namespace = generate_train_data_2d_triangle_TFR.parse_args()
    print(arg_namespace)
    arg_namespace.to_log_file = True
    arg_namespace.mode = "val"
    arg_namespace.data_id = "triangle2d_32_dummy"
    print(arg_namespace)
    generate_train_data_2d_triangle_TFR.main(arg_namespace)
