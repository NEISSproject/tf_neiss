import tensorflow as tf


class ScatterPolygonTF(tf.keras.layers.Layer):
    def __init__(self, fc_tensor, points_tf=tf.constant([[0, 0], [0, 1], [1, 0]]),
                 epsilon=tf.constant(0.0001), with_batch_dim=True, dtype=tf.float64):
        tf.keras.layers.Layer.__init__(self, trainable=False)
        self._with_batch_dim_tf = with_batch_dim
        self.mydtype = dtype
        if self.mydtype == tf.float64:
            self.complex_dtype=tf.complex128
        elif self.mydtype == tf.float32:
            self.complex_dtype=tf.complex64
        else:
            raise TypeError("only tf.float32 and tf.float64 are supported! Found: {}".format(self.mydtype))
        if self._with_batch_dim_tf:
            self.phi_tf = tf.constant(fc_tensor[0, 0, :], dtype=self.mydtype)
        else:
            self.phi_tf = tf.constant(fc_tensor[0, :], dtype=self.mydtype)
        self._q_tf = self.q_of_phi(self.phi_tf)

        self.epsilon_tf = tf.cast(epsilon, dtype=self.mydtype)
        self.points_tf = tf.cast(points_tf, dtype=self.mydtype)
        self._cross = tf.constant([[0.0, -1.0], [1.0, 0.0]], dtype=self.mydtype)

    def __call__(self, points_tf, *args, **kwargs):
        self.update_points(points_tf)
        return self.fc_of_phi()

    def update_points(self, points_tf):
        self.points_tf = tf.cast(points_tf, dtype=self.mydtype)

    def update_phi_array(self, phi_array):
        self.phi_tf = tf.cast(phi_array, dtype=self.mydtype)

    def q_of_phi(self, phi_tf):
        a__tf = tf.math.cos(phi_tf)
        b__tf = tf.math.sin(phi_tf) - 1.0
        q_tf = tf.stack([a__tf, b__tf])
        return q_tf

    def fc_of_edge(self, p0_tf, p1_tf, c=0.0):
        j_tf = tf.cast(tf.complex(0.0, 1.0), dtype=self.complex_dtype)

        c_tfc = tf.cast(c, dtype=self.complex_dtype)

        q_cross_tf = tf.matmul(self._cross, self._q_tf)

        p0p1_tf = p1_tf - p0_tf
        if self._with_batch_dim_tf:
            scale_tf = tf.cast(1.0 / tf.math.abs(self._q_tf[0] ** 2 + self._q_tf[1] ** 2), dtype=self.complex_dtype)
            f_p0_tf = -tf.cast(1.0, dtype=self.complex_dtype) * tf.math.exp(j_tf * (self.batch_complex_dot(p0_tf, self._q_tf) + c_tfc))
            f_p1_tf = -tf.cast(1.0, dtype=self.complex_dtype) * tf.math.exp(j_tf * (self.batch_complex_dot(p1_tf, self._q_tf) + c_tfc))
            case1_array_tf = scale_tf * self.batch_complex_dot(p0p1_tf, q_cross_tf) * (f_p1_tf - f_p0_tf) / self.batch_complex_dot(p0p1_tf,
                                                                                                             self._q_tf)
            case2_array_tf = scale_tf * self.batch_complex_dot(p0p1_tf, q_cross_tf) * -j_tf * tf.math.exp(
                j_tf * self.batch_complex_dot(p0_tf, self._q_tf) + c_tfc)
            res_array_tf = tf.where(tf.math.abs(self.batch_complex_dot(p0p1_tf, self._q_tf)) >= 0.0001, case1_array_tf, case2_array_tf)
        else:
            scale_tf = tf.cast(1.0 / tf.math.abs(self._q_tf[0] ** 2 + self._q_tf[1] ** 2), dtype=self.complex_dtype)
            f_p0_tf = -tf.cast(1.0, dtype=self.complex_dtype) * tf.math.exp(j_tf * (self.complex_dot(p0_tf, self._q_tf) + c_tfc))
            f_p1_tf = -tf.cast(1.0, dtype=self.complex_dtype) * tf.math.exp(j_tf * (self.complex_dot(p1_tf, self._q_tf) + c_tfc))
            case1_array_tf = scale_tf * self.complex_dot(p0p1_tf, q_cross_tf) * (f_p1_tf - f_p0_tf) / self.complex_dot(p0p1_tf,
                                                                                                             self._q_tf)
            case2_array_tf = scale_tf * self.complex_dot(p0p1_tf, q_cross_tf) * -j_tf * tf.math.exp(
                j_tf * self.complex_dot(p0_tf, self._q_tf) + c_tfc)
            res_array_tf = tf.where(tf.math.abs(self.complex_dot(p0p1_tf, self._q_tf)) >= 0.0001, case1_array_tf, case2_array_tf)

        return res_array_tf

    def fc_of_phi(self):
        c = tf.cast(0.0, dtype=self.mydtype)
        sum_res = tf.zeros_like(self.phi_tf, dtype=self.complex_dtype)
        if self._with_batch_dim_tf:
            sum_res = tf.expand_dims(sum_res, axis=0)
            point_dim = self.points_tf.shape[1]
            for index in range(point_dim):
                p0 = self.points_tf[:, index - 1]
                p1 = self.points_tf[:, index]
                sum_res += tf.expand_dims(self.fc_of_edge(p0, p1, c=c), axis=1)
        else:
            point_dim = self.points_tf.shape[0]
            for index in range(point_dim):
                p0 = self.points_tf[index - 1]
                p1 = self.points_tf[index]
                sum_res += tf.expand_dims(self.fc_of_edge(p0, p1, c=c), axis=0)
        if self._with_batch_dim_tf:
            res = tf.concat((tf.math.real(sum_res), tf.math.imag(sum_res)), axis=1)
        else:
            res = tf.concat((tf.math.real(sum_res), tf.math.imag(sum_res)), axis=0)

        return res

    def complex_dot(self, a, b):
        return tf.cast(tf.einsum('...i,i...->...', a, b), dtype=self.complex_dtype)

    def batch_complex_dot(self, a, b):
        return tf.cast(tf.einsum('j...i,i...->j...', a, b), dtype=self.complex_dtype)

