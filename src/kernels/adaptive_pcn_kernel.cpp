#include "kernels/adaptive_pcn_kernel.hpp"
#include "pcn.hpp"
#include "kernel_result.hpp"
#include <memory>
#include "logging/logging.hpp"

AdaptivePCNKernel::AdaptivePCNKernel(
		double beta,
		std::function<double(const Eigen::VectorXd &)> log_likelihood_function, 
		const Eigen::VectorXd prior_mean,
		const Eigen::MatrixXd sqrt_prior_cov
	) : _beta(beta), _log_likelihood_function(log_likelihood_function), _prior_mean(prior_mean), _sqrt_prior_cov(sqrt_prior_cov)
{ }

std::unique_ptr<OneKernelResult> AdaptivePCNKernel::apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path) {
	auto ret = apply_one_kernel_pcn(sample, n_transitions, _beta, _log_likelihood_function, _prior_mean, _sqrt_prior_cov, return_path);

	_cur_accept_rate = (_cur_transitions*_cur_accept_rate + n_transitions*ret->average_acceptance) / (_cur_transitions + n_transitions);
	_cur_transitions += n_transitions;

	return ret;
}

void AdaptivePCNKernel::adapt() {
	const double ADAPT_FACTOR = 1.25;
	if(_cur_accept_rate < 0.1) {
		_beta /= ADAPT_FACTOR;
		LOG_INFO(label << " adapted to " << _beta << " due to accept rate " << _cur_accept_rate);
	} 
	else if(_cur_accept_rate > 0.4 && _beta < 1.0) {
		_beta *= ADAPT_FACTOR;
		LOG_INFO(label << " adapted to " << _beta << " due to accept rate " << _cur_accept_rate);
	}
	/*
	else
		LOG_INFO(label << " needed no adaptation, accept was " << _cur_accept_rate);
	*/

	if(_beta > 1.0) {
		_beta = 1.0;	
	}
	_cur_transitions = 0;
	_cur_accept_rate = 0.0;
}

double AdaptivePCNKernel::log_target(const Eigen::VectorXd &sample) const {
	return _log_likelihood_function(sample);
}