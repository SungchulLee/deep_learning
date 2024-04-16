from global_name_space import ARGS

def train(theta):
    theta_trace = [theta]
    loss_trace = [ARGS.compute_loss(theta)]
    for i in range(ARGS.epochs):
        grad = ARGS.compute_gradient(theta_trace[-1])
        theta = ARGS.apply_gradient_descent(theta_trace[-1], grad, ARGS.lr)
        theta_trace.append(theta)
        loss_trace.append(ARGS.compute_loss(theta))
    return theta, theta_trace, loss_trace