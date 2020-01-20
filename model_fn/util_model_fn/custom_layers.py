import tensorflow as tf


class ScatterPolygonTF(tf.keras.layers.Layer):
    def __init__(self, fc_tensor, points_tf=tf.constant([[0, 0], [0, 1], [1, 0]]),
                 epsilon=tf.constant(0.0001, dtype=tf.float64), with_batch_dim=True):
        tf.keras.layers.Layer.__init__(self, trainable=False)
        self._with_batch_dim_tf = tf.constant(with_batch_dim, dtype=tf.bool)
        if self._with_batch_dim_tf:
            self.phi_tf = tf.constant(fc_tensor[0, 0, :], dtype=tf.float64)
        else:
            self.phi_tf = tf.constant(fc_tensor[0, :], dtype=tf.float64)
        self.epsilon_tf = epsilon
        self.points_tf = points_tf
        self._cross = tf.constant([[0.0, -1.0], [1.0, 0.0]], dtype=tf.float64)
        self._q_tf = self.q_of_phi(self.phi_tf)

    def __call__(self, points_tf, *args, **kwargs):
        self.update_points(points_tf)
        return self.fc_of_phi()

    def update_points(self, points_tf):
        self.points_tf = points_tf

    def q_of_phi(self, phi_tf):
        a__tf = tf.math.cos(phi_tf)
        b__tf = tf.math.sin(phi_tf) - 1.0
        if self._with_batch_dim_tf:
            q_tf = tf.stack([a__tf, b__tf], axis=1)
        else:
            q_tf = tf.stack([a__tf, b__tf])
        return q_tf

    def fc_of_edge(self, p0_tf, p1_tf, c=0.0):
        j_tf = tf.cast(tf.complex(0.0, 1.0), dtype=tf.complex128)

        c_tfc = tf.cast(c, dtype=tf.complex128)

        q_cross_tf = tf.matmul(self._cross, self._q_tf)

        p0p1_tf = p1_tf - p0_tf
        scale_tf = tf.cast(1.0 / tf.math.abs(self._q_tf[0] ** 2 + self._q_tf[1] ** 2), dtype=tf.complex128)
        f_p0_tf = -tf.cast(1.0, dtype=tf.complex128) * tf.math.exp(j_tf * (complex_dot(p0_tf, self._q_tf) + c_tfc))
        f_p1_tf = -tf.cast(1.0, dtype=tf.complex128) * tf.math.exp(j_tf * (complex_dot(p1_tf, self._q_tf) + c_tfc))
        case1_array_tf = scale_tf * complex_dot(p0p1_tf, q_cross_tf) * (f_p1_tf - f_p0_tf) / complex_dot(p0p1_tf,
                                                                                                         self._q_tf)
        case2_array_tf = scale_tf * complex_dot(p0p1_tf, q_cross_tf) * -j_tf * tf.math.exp(
            j_tf * complex_dot(p0_tf, self._q_tf) + c_tfc)
        res_array_tf = tf.where(tf.math.abs(complex_dot(p0p1_tf, self._q_tf)) >= 0.0001, case1_array_tf, case2_array_tf)

        return res_array_tf

    @tf.function
    def fc_of_phi(self):
        c = tf.cast(0.0, dtype=tf.float64)
        sum_res = tf.zeros_like(self.phi_tf, dtype=tf.complex128)
        for index in range(self.points_tf.shape[0]):
            p0 = self.points_tf[index - 1]
            p1 = self.points_tf[index]
            sum_res += self.fc_of_edge(p0, p1, c=c)

        return sum_res


def complex_dot(a, b):
    return tf.cast(tf.einsum('i,i...->...', a, b), dtype=tf.complex128)
