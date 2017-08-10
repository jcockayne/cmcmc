#include "kernels/base.hpp"
#include <memory>

#ifndef ADAPTIVE_PCN_KERNEL_H

class AdaptivePCNKernel : public TransitionKernel {
public:
	AdaptivePCNKernel(
		double beta,
		std::function<double(const Eigen::VectorXd &)> log_likelihood_function, 
		const Eigen::VectorXd prior_mean,
		const Eigen::MatrixXd sqrt_prior_cov
	);
	virtual std::unique_ptr<OneKernelResult> apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path=true);
	virtual double log_target(const Eigen::VectorXd &sample) const;
	virtual void adapt();
private:
	double _beta;
	std::function<double(const Eigen::VectorXd &)> _log_likelihood_function;
	const Eigen::VectorXd _prior_mean;
	const Eigen::MatrixXd _sqrt_prior_cov;
	double _cur_accept_rate = 0.0;
	int _cur_transitions = 0;
};

#define ADAPTIVE_PCN_KERNEL_H
#endif